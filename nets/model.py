import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class AGDH_Module(nn.Module):
    """
    双路径高阶动态超图 (Dual-path AGDH)
    路径 1: Window-based KNN (捕捉局部精细拓扑，隐式 H 矩阵防显存爆炸)
    路径 2: Anatomy-Guided Clustering (捕捉全局宏观解剖学先验)
    """
    def __init__(self, in_channels, k=8, window_size=(4, 4, 4), num_anatomy_clusters=4):
        super().__init__()
        self.k = k
        self.ws = window_size
        self.d = in_channels
        
        # ==========================================
        # 路径 1: 局部 KNN 边更新网络
        # ==========================================
        self.edge_update_knn = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.InstanceNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels)
        )
        
        # ==========================================
        # 路径 2: 全局解剖学聚类 (Anatomy-Guided Path)
        # ==========================================
        # 聚类投影：将特征映射到对应的解剖结构 (如 股骨、胫骨及其软骨)
        self.cluster_proj = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, num_anatomy_clusters)
        )
        # 解剖学边更新网络
        self.edge_update_anatomy = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.InstanceNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels)
        )
        
        # ==========================================
        # 融合与门控模块
        # ==========================================
        self.gate_attention = nn.Sequential(
            nn.Linear(in_channels * 2, 1),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x, D, H, W):
        """
        x: (B, N, C)
        D, H, W: 原始特征图尺寸
        """
        B, N, C = x.shape
        residual = x
        
        # ---------------------------------------------------------
        # Path 2: Anatomy-Guided Path (全局解剖学聚类)
        # 对应图左下角: Anatomy-Guided Path (Prior-injected)
        # ---------------------------------------------------------
        # 1. 节点到超边聚合 (Node-to-Edge Aggregation)
        # 计算每个节点属于哪个解剖学聚类的概率
        cluster_probs = F.softmax(self.cluster_proj(x), dim=-1) # (B, N, num_clusters)
        
        # 将节点特征聚合为宏观解剖学超边特征
        # bmm: (B, num_clusters, N) x (B, N, C) -> (B, num_clusters, C)
        anatomy_edges = torch.bmm(cluster_probs.transpose(1, 2), x) 
        # 归一化，防止某些聚类为空导致数值不稳定
        anatomy_edges = anatomy_edges / (cluster_probs.sum(dim=1, keepdim=True).transpose(1, 2) + 1e-5)
        
        # 2. 超边特征更新 (Edge Feature Update)
        anatomy_edges = self.edge_update_anatomy(anatomy_edges)
        
        # 3. 超边到节点分布 (Edge-to-Node Distribution)
        # 将解剖学超边特征重新分配给各个节点
        x_anatomy = torch.bmm(cluster_probs, anatomy_edges) # (B, N, C)
        
        # ---------------------------------------------------------
        # Path 1: Window-based KNN (局部精细拓扑建图)
        # 对应图左上角: KNN Graph Construction
        # ---------------------------------------------------------
        x_img = x.transpose(1, 2).view(B, C, D, H, W)
        
        # Padding 逻辑
        pad_d = (self.ws[0] - D % self.ws[0]) % self.ws[0]
        pad_h = (self.ws[1] - H % self.ws[1]) % self.ws[1]
        pad_w = (self.ws[2] - W % self.ws[2]) % self.ws[2]
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x_img = F.pad(x_img, (0, pad_w, 0, pad_h, 0, pad_d))
        
        _, _, pD, pH, pW = x_img.shape
        
        # 切分窗口: (B, C, D, H, W) -> (B, nW, win_vol, C)
        x_win = rearrange(x_img, 'b c (d w1) (h w2) (w w3) -> (b d h w) (w1 w2 w3) c', 
                          w1=self.ws[0], w2=self.ws[1], w3=self.ws[2])
        
        B_win, N_win, _ = x_win.shape
        
        # 窗口内 KNN
        dist = torch.cdist(x_win, x_win)
        _, knn_idx = dist.topk(k=min(self.k, N_win), dim=-1, largest=False)
        
        # 节点到超边 (KNN)
        idx_base = torch.arange(0, B_win, device=x.device).view(-1, 1, 1) * N_win
        flat_idx = (knn_idx + idx_base).view(-1)
        neighbor_feats = x_win.view(-1, C)[flat_idx].view(B_win, N_win, -1, C)
        
        edge_feats_knn = neighbor_feats.mean(dim=2)
        edge_feats_knn = self.edge_update_knn(edge_feats_knn) # 超边特征更新
        
        # 还原回原始形状
        out_img = rearrange(edge_feats_knn, '(b d h w) (w1 w2 w3) c -> b c (d w1) (h w2) (w w3)', 
                            b=B, d=pD//self.ws[0], h=pH//self.ws[1], w=pW//self.ws[2],
                            w1=self.ws[0], w2=self.ws[1], w3=self.ws[2])
        
        out_img = out_img[:, :, :D, :H, :W]
        x_knn = out_img.reshape(B, C, -1).transpose(1, 2) # (B, N, C)
        
        # ---------------------------------------------------------
        # Dual-path Fusion & Gated Residual (双路融合与门控残差)
        # 对应图最右侧
        # ---------------------------------------------------------
        # 将局部 KNN 拓扑特征与全局解剖学特征相加融合
        refined_features = x_knn + x_anatomy 
        
        gate_input = torch.cat([refined_features, x], dim=-1)
        gate = self.gate_attention(gate_input)
        
        out = gate * refined_features + (1 - gate) * x
        
        return self.norm(out + residual)

class HyperNet(nn.Module):
    def __init__(self, channels, k=8, window_size=(4, 4, 4), num_anatomy_clusters=4):
        super().__init__()
        self.agdh = AGDH_Module(channels, k=k, window_size=window_size, num_anatomy_clusters=num_anatomy_clusters)
        
    def forward(self, x):
        if len(x.shape) == 5:
            B, C, D, H, W = x.shape
            x_flat = x.view(B, C, -1).permute(0, 2, 1).contiguous()
            out_flat = self.agdh(x_flat, D, H, W)
            out = out_flat.permute(0, 2, 1).view(B, C, D, H, W).contiguous()
            return out
        elif len(x.shape) == 3:
            raise ValueError(f"HyperNet 接收到了 3D 输入 {x.shape}，请在主网络调用时确保传入 5D 张量。")
        return x