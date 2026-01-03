# model.py - 只预测 Yaw 角度版本
import torch
import torch.nn as nn
import timm
import config


class PalletPoseEstimator(nn.Module):
    """
    托盘位姿估计模型（RGB 输入，预测位置 + Yaw角）
    
    输入：RGB 图像 (3, 224, 224)
    输出：
        - position: (x, y, z) 位置，单位：米
        - yaw: 绕 Z 轴旋转角，单位：弧度
    
    注意：Roll 和 Pitch 假设恒为 0
    """
    
    def __init__(self, pretrained=config.PRETRAINED):
        super(PalletPoseEstimator, self).__init__()
        
        # ============ RGB 骨干网络 ============
        self.backbone = timm.create_model(
            'efficientnet_b3',
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
            drop_rate=0.3,
            drop_path_rate=0.2
        )
        
        feature_dim = self.backbone.num_features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # ============ 位置回归头 ============
        self.position_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, 3)  # 输出 (x, y, z)
        )
        
        # ============ Yaw 角回归头 ============
        self.rotation_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, 1)  # 只输出 1 维（Yaw）
        )
    
    def forward(self, rgb):
        """
        Args:
            rgb: (B, 3, H, W) RGB 图像
        
        Returns:
            position: (B, 3) 位置 (x, y, z)
            yaw: (B, 1) Yaw 角度（弧度）
        """
        features = self.backbone(rgb)
        features = self.global_pool(features)
        features = features.flatten(1)
        
        position = self.position_head(features)  # (B, 3)
        yaw = self.rotation_head(features)       # (B, 1)
        
        return position, yaw
    
    def get_num_params(self):
        """返回模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PalletPoseLoss(nn.Module):
    """
    位姿估计损失函数
    
    包含：
    - 位置损失（MSE）
    - Yaw 角度损失（考虑周期性）
    """
    
    def __init__(self, pos_weight=1.0, rot_weight=1.0, z_weight=10.0):
        super(PalletPoseLoss, self).__init__()
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight
        self.z_weight = z_weight  # Z 轴额外权重
        self.mse = nn.MSELoss()
    
    def forward(self, position_pred, position_gt, yaw_pred, yaw_gt):
        """
        Args:
            position_pred: (B, 3) 预测位置
            position_gt: (B, 3) 真实位置
            yaw_pred: (B, 1) 预测 Yaw 角
            yaw_gt: (B, 1) 真实 Yaw 角
        
        Returns:
            loss: 总损失
            pos_loss: 位置损失
            rot_loss: Yaw 损失
        """
        # XY 损失
        xy_loss = self.mse(position_pred[:, :2], position_gt[:, :2])
        
        # Z 损失（加权）
        z_loss = self.mse(position_pred[:, 2], position_gt[:, 2]) * self.z_weight
        
        pos_loss = xy_loss + z_loss
        
        # Yaw 损失
        yaw_diff = yaw_pred - yaw_gt
        yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
        rot_loss = torch.mean(yaw_diff ** 2)
        
        loss = self.pos_weight * pos_loss + self.rot_weight * rot_loss
        
        return loss, pos_loss, rot_loss


if __name__ == '__main__':
    # 测试模型
    print("Testing PalletPoseEstimator (Yaw-only version)...")
    print("=" * 60)
    
    model = PalletPoseEstimator(pretrained=False)
    
    # 测试输入
    batch_size = 4
    rgb = torch.randn(batch_size, 3, 224, 224)
    
    # 前向传播
    position, yaw = model(rgb)
    
    print(f"Input RGB shape:     {rgb.shape}")
    print(f"Output position:     {position.shape}")
    print(f"Output yaw:          {yaw.shape}")
    print(f"\nModel parameters:    {model.get_num_params():,}")
    
    # 测试损失函数
    print("\nTesting PalletPoseLoss...")
    criterion = PalletPoseLoss(pos_weight=1.0, rot_weight=1.0)
    
    position_gt = torch.randn(batch_size, 3)
    yaw_gt = torch.randn(batch_size, 1)
    
    loss, pos_loss, rot_loss = criterion(position, position_gt, yaw, yaw_gt)
    
    print(f"Total loss:          {loss.item():.6f}")
    print(f"Position loss:       {pos_loss.item():.6f}")
    print(f"Yaw loss:            {rot_loss.item():.6f}")
    
    print("=" * 60)
    print("✓ Model test passed!")
