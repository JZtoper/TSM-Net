# nnUNetTrainerUMambaBot.py

import os
import sys

# 获取 nnunetv2 根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
nnunetv2_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if nnunetv2_root not in sys.path:
    sys.path.insert(0, nnunetv2_root)

# 改成从 nnunetv2 根开始导入
from nnunetv2.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.network_training.nnUNet_variants.architectures.generic_modular_UNet import Generic_UNet

class nnUNetTrainerUMambaBot(nnUNetTrainer):
    """
    自定义 Trainer，用于在 nnU-Net 框架下训练 UMambaBot 3D 网络
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 如果有自定义网络，可以在这里替换 self.network
        # self.network = UMambaBot3D(...)

        # 自定义超参数
        self.initial_lr = 1e-3
        self.num_epochs = 500
        self.batch_size = 2
        self.deep_supervision = True

        # 你可以在这里添加其他自定义属性
        self.custom_flag = True

    @staticmethod
    def build_network_architecture(plans_manager, dataset_json, configuration_manager,
                                   num_input_channels, enable_deep_supervision=True):
        """
        构建 UMambaBot 3D 网络结构
        """
        # 示例：使用 nnU-Net 默认的 Generic_UNet 构建网络
        # 如果你有自己的 UMambaBot3D 网络，请替换下面这行
        network = Generic_UNet(
            input_channels=num_input_channels,
            base_num_features=30,
            num_classes=plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            num_pool_per_axis=[5, 5, 5],  # 可根据你的数据修改
            conv_per_stage=2,
            feat_map_mul_on_downscale=2,
            conv_op=torch.nn.Conv3d,
            norm_op=torch.nn.BatchNorm3d,
            dropout_op=torch.nn.Dropout3d,
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={'negative_slope': 0.01, 'inplace': True},
            deep_supervision=enable_deep_supervision,
            upscale_logits=False,
            initial_stride=[1,2,2],
        )
        return network

    def initialize(self, training=True, force_load_plans=False):
        """
        初始化训练器
        """
        super().initialize(training, force_load_plans)
        # 可在这里加入自定义初始化操作
        if self.custom_flag:
            print("UMambaBot Trainer 初始化完成")

    def run_training(self):
        """
        开始训练
        """
        print("开始 UMambaBot 训练")
        super().run_training()

    def predict_preprocessed_data_return_softmax(self, data, do_mirroring=True, use_sliding_window=True,
                                                 step=0.5, patch_size=None, batch_size=1, 
                                                 mirror_axes=(0, 1, 2), use_gaussian=True, verbose=True):
        """
        自定义预测函数，可以直接使用 nnU-Net 默认预测逻辑
        """
        return super().predict_preprocessed_data_return_softmax(
            data, do_mirroring, use_sliding_window, step, patch_size,
            batch_size, mirror_axes, use_gaussian, verbose
        )

# 下面可以写一个测试入口
if __name__ == "__main__":
    from nnunet.paths import default_plans_identifier
    from nnunet.dataset_conversion.utils import generate_dataset_json

    # 示例参数，请替换为你的路径
    plans_file = "F:/OAI/nnUNetPlans/nnUNetPlansv2.1_plans.pkl"
    fold = 0
    dataset_json = "F:/OAI/dataset.json"

    trainer = nnUNetTrainerUMambaBot(plans_file, fold, dataset_json=dataset_json)
    trainer.initialize(True)
    trainer.run_training()
