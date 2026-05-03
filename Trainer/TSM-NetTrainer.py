import os
import sys
import torch
import numpy as np
from torch import autocast
from torch.nn.utils import clip_grad_norm_

# 确保路径能找到 umamba 模块
project_root = '/root/autodl-tmp/U-Mamba'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from umamba.nnunetv2.nets.UMambaBot_3d import UMambaBot3D
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss


class nnUNetTrainerUMambaBot(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        os.environ["OMP_NUM_THREADS"] = "1"
        
        # 1. 禁用深监督
        self.enable_deep_supervision = False
        
        # 2. 使用保守的学习率
        self.initial_lr = 1e-4
        
        # 3. 获取通道和类别信息 (供训练时实例方法使用)
        self.num_input_channels = 1
        self.num_classes = len(dataset_json['labels'])
        
        # 4. 梯度累积与迭代控制
        self.gradient_accumulation_steps = 1
        self.debug_mode = True
        self.current_iteration = 0
        self.gradient_clip_val = 1.0
        
        # 5. 清空CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    @staticmethod
    def build_network_architecture(plans_manager, dataset_json, configuration_manager, 
                                   num_input_channels, enable_deep_supervision: bool = True):
        """构建模型实例 - 支持超图模块消融控制"""
        
        # 安全获取类别数量
        num_classes = len(dataset_json['labels'].keys())
        
        # --- 消融实验核心控制开关 ---
        use_vss_flag = True    # 保持 VSS 开启
        use_hyper_flag = True  # <--- 修改这里进行超图消融：True 为完整版，False 为消融版
        base_ch_val = 16       # 显存优化值
        
        print(f"\n{'='*50}")
        print(f"Initializing UMambaBot3D (HGMamba-Net)")
        print(f"VSS Module: {'Enabled' if use_vss_flag else 'DISABLED'}")
        print(f"HyperNet Module: {'Enabled' if use_hyper_flag else 'DISABLED (Ablation Mode)'}")
        print(f"Input channels: {num_input_channels}")
        print(f"Number of classes: {num_classes}")
        print(f"{'='*50}\n")
        
        # 实例化模型，传入 use_hyper 参数
        model = UMambaBot3D(
            in_channels=num_input_channels,
            num_classes=num_classes,
            base_ch=base_ch_val,
            use_vss=use_vss_flag,
            use_hyper=use_hyper_flag  # 确保你的网络类接收此参数
        )
        
        return model

    def build_loss(self):
        """构建稳定的损失函数"""
        print("Building loss function: DC_and_CE_loss...")
        loss = DC_and_CE_loss(
            {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False},
            {'label_smoothing': 0.1}
        )
        return loss

    def set_deep_supervision_enabled(self, enabled: bool):
        # 覆盖父类方法以确保推理时不触发多尺度输出逻辑
        pass

    def configure_optimizers(self):
        print(f"Configuring optimizer: AdamW (lr={self.initial_lr}, weight_decay=1e-2)")
        optimizer = torch.optim.AdamW(
            self.network.parameters(), 
            lr=self.initial_lr, 
            weight_decay=1e-2,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def train_step(self, batch: dict) -> dict:
        self.current_iteration += 1
        
        if self.debug_mode and self.current_iteration == 1:
            if torch.cuda.is_available():
                print(f"\nInitial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        data = batch['data']
        target = batch['target'] if isinstance(batch['target'], torch.Tensor) else batch['target'][0]
        
        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        
        data = torch.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
        
        self.optimizer.zero_grad()
        
        with autocast(self.device.type, enabled=True):
            try:
                output = self.network(data)
                
                if isinstance(output, (list, tuple)):
                    output = output[0]
                
                output = torch.clamp(output, -50, 50)
                loss = self.loss(output, target)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Loss is NaN/Inf at iteration {self.current_iteration}")
                    return {'loss': 1.0}
                    
            except RuntimeError as e:
                print(f"CUDA error in forward pass: {e}")
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    dummy_loss = self.network.parameters().__next__().sum() * 0.0
                return {'loss': dummy_loss + 1.0}
        
        self.grad_scaler.scale(loss).backward()
        
        self.grad_scaler.unscale_(self.optimizer)
        clip_grad_norm_(self.network.parameters(), max_norm=self.gradient_clip_val)
        
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        
        return {'loss': loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target'] if isinstance(batch['target'], torch.Tensor) else batch['target'][0]
        target = target.to(self.device, non_blocking=True)
        
        data = torch.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
        
        with autocast(self.device.type, enabled=True):
            try:
                output = self.network(data)
                if isinstance(output, (list, tuple)):
                    output = output[0]
                
                output = torch.clamp(output, -50, 50)
                loss = self.loss(output, target)
            except RuntimeError as e:
                print(f"Error in validation: {e}")
                loss = torch.tensor(1.0, device=self.device)
                output = torch.zeros((data.shape[0], self.num_classes, *data.shape[2:]), device=self.device)

        output_seg = output.argmax(1, keepdim=True)
        axes = tuple(range(2, len(output.shape)))
        
        tp_list, fp_list, fn_list = [], [], []
        for c in range(1, self.num_classes):
            p = (output_seg == c)
            t = (target == c)
            tp_list.append(torch.sum(p & t, dim=axes))
            fp_list.append(torch.sum(p & (~t), dim=axes))
            fn_list.append(torch.sum((~p) & t, dim=axes))

        if tp_list:
            tp_hard = torch.stack(tp_list, dim=1).sum(0).detach().cpu().numpy()
            fp_hard = torch.stack(fp_list, dim=1).sum(0).detach().cpu().numpy()
            fn_hard = torch.stack(fn_list, dim=1).sum(0).detach().cpu().numpy()
        else:
            tp_hard = fp_hard = fn_hard = np.array([])

        return {
            'loss': loss.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
        }
