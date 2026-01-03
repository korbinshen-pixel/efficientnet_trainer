# dataset.py - 直接读取欧拉角版本
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
from sklearn.model_selection import train_test_split
import config


def get_latest_dataset_dir(root_dir=config.DATASET_ROOT):
    """获取最新采集的数据集目录"""
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Dataset root dir not found: {root_dir}")
    
    # 找所有时间戳目录（格式：20251210_162120）
    subdirs = [d for d in os.listdir(root_dir) 
               if os.path.isdir(os.path.join(root_dir, d))]
    
    if not subdirs:
        raise FileNotFoundError(f"No dataset directories found in {root_dir}")
    
    subdirs.sort()  # 按字母序（时间戳自然排序）
    latest = os.path.join(root_dir, subdirs[-1])
    print(f"Using latest dataset: {latest}")
    return latest


class PalletPoseDataset(Dataset):
    """
    托盘位姿数据集加载器（仅RGB + 欧拉角）
    
    期望的目录结构：
        dataset_dir/
        ├── rgb/
        │   ├── rgb_000000.png
        │   ├── rgb_000001.png
        │   └── ...
        └── poses.txt  (每行：序号 x y z roll pitch yaw)
    
    注意：序号仅用于索引对应的rgb图像，训练时不使用
    """
    
    def __init__(self, dataset_dir, split='train', transform_rgb=None, augmentation=False):
        """
        Args:
            dataset_dir: 数据集目录
            split: 'train', 'val', 或 'test'
            transform_rgb: RGB 图像的数据增强/归一化
            augmentation: 是否启用数据增强
        """
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform_rgb = transform_rgb
        self.augmentation = augmentation
        
        # 加载 RGB 文件列表
        self.rgb_dir = os.path.join(dataset_dir, 'rgb')
        self.pose_file = os.path.join(dataset_dir, 'poses.txt')
        
        if not os.path.exists(self.rgb_dir):
            raise FileNotFoundError(f"RGB dir not found: {self.rgb_dir}")
        if not os.path.exists(self.pose_file):
            raise FileNotFoundError(f"Pose file not found: {self.pose_file}")
        
        # 读取 RGB 文件列表（按名字排序）
        self.rgb_files = sorted(glob.glob(os.path.join(self.rgb_dir, 'rgb_*.png')))
        
        # 读取位姿数据（直接是欧拉角格式）
        self.poses = self._load_poses()
        
        # 确保数据集大小一致
        assert len(self.rgb_files) == len(self.poses), \
            f"RGB and Pose counts mismatch: {len(self.rgb_files)} vs {len(self.poses)}"
        
        print(f"Loaded {len(self.rgb_files)} samples from {dataset_dir}")
        
        # 分割数据集
        indices = np.arange(len(self.rgb_files))
        
        train_idx, test_idx = train_test_split(
            indices, test_size=(1 - config.TRAIN_SPLIT), random_state=42
        )
        val_idx, test_idx = train_test_split(
            test_idx, test_size=config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT),
            random_state=42
        )
        
        if split == 'train':
            self.indices = train_idx
        elif split == 'val':
            self.indices = val_idx
        elif split == 'test':
            self.indices = test_idx
        else:
            raise ValueError(f"Unknown split: {split}")
        
        print(f"Split '{split}': {len(self.indices)} samples")
    
    def _load_poses(self):
        """
        加载 poses.txt，格式：序号 x y z roll pitch yaw
        返回：{序号: (position, euler)} 字典
        """
        poses = {}
        with open(self.pose_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 7:
                    print(f"Warning: Skipping invalid line: {line.strip()}")
                    continue
                
                frame_id = parts[0]  # 序号（字符串）
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                roll, pitch, yaw = float(parts[4]), float(parts[5]), float(parts[6])
                
                position = np.array([x, y, z], dtype=np.float32)
                euler = np.array([roll, pitch, yaw], dtype=np.float32)
                
                poses[frame_id] = (position, euler)
        
        return poses
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """返回 (rgb, position, rotation_euler)"""
        actual_idx = self.indices[idx]
        
        # 加载 RGB 图像
        rgb_path = self.rgb_files[actual_idx]
        rgb = Image.open(rgb_path).convert('RGB')
        
        # 提取位姿（从文件名提取序号）
        # 文件名格式: rgb_000000.png -> 序号: 000000
        frame_id = os.path.basename(rgb_path).replace('rgb_', '').replace('.png', '')
        
        if frame_id not in self.poses:
            raise KeyError(f"Frame ID {frame_id} not found in poses.txt")
        
        position, euler = self.poses[frame_id]
        
        # 数据增强（可选）
        if self.augmentation and self.split == 'train':
            rgb = self._augment_rgb(rgb)
        
        # 转换为张量
        if self.transform_rgb:
            rgb = self.transform_rgb(rgb)
        else:
            rgb = transforms.ToTensor()(rgb)
        
        # 返回数据
        return {
            'rgb': rgb,                              # (3, H, W)
            'position': torch.from_numpy(position),  # (3,) x, y, z
            'rotation': torch.from_numpy(euler),     # (3,) roll, pitch, yaw (欧拉角，单位取决于你的数据)
            'frame_id': frame_id
        }
    
    def _augment_rgb(self, rgb):
        """RGB 图像增强"""
        # 随机亮度调整
        if config.USE_AUGMENTATION and np.random.rand() > 0.5:
            from PIL import ImageEnhance
            brightness_factor = 1 + np.random.uniform(
                -config.AUGMENT_RGB_BRIGHTNESS, 
                config.AUGMENT_RGB_BRIGHTNESS
            )
            enhancer = ImageEnhance.Brightness(rgb)
            rgb = enhancer.enhance(brightness_factor)
        
        # 随机对比度调整
        if config.USE_AUGMENTATION and np.random.rand() > 0.5:
            from PIL import ImageEnhance
            contrast_factor = 1 + np.random.uniform(
                -config.AUGMENT_RGB_CONTRAST, 
                config.AUGMENT_RGB_CONTRAST
            )
            enhancer = ImageEnhance.Contrast(rgb)
            rgb = enhancer.enhance(contrast_factor)
        
        # 随机水平翻转（可选，看你的场景是否对称）
        # if config.USE_AUGMENTATION and np.random.rand() > 0.5:
        #     rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
        
        return rgb


def get_dataloaders(dataset_dir=None, batch_size=config.BATCH_SIZE):
    """
    获取 train/val/test 数据加载器
    
    Args:
        dataset_dir: 数据集目录，如果为 None 则自动找最新的
        batch_size: 批大小
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    if dataset_dir is None:
        dataset_dir = get_latest_dataset_dir()
    
    # RGB 图像的标准化（ImageNet 统计）
    rgb_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # EfficientNet-B3 推荐输入大小
        transforms.ToTensor(),
        transforms.Normalize(mean=config.RGB_MEAN, std=config.RGB_STD)
    ])
    
    # 创建三个数据集
    train_dataset = PalletPoseDataset(
        dataset_dir, split='train',
        transform_rgb=rgb_transform,
        augmentation=config.USE_AUGMENTATION
    )
    
    val_dataset = PalletPoseDataset(
        dataset_dir, split='val',
        transform_rgb=rgb_transform,
        augmentation=False
    )
    
    test_dataset = PalletPoseDataset(
        dataset_dir, split='test',
        transform_rgb=rgb_transform,
        augmentation=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 测试数据加载器
    print("Testing DataLoader (RGB only + Euler angles)...")
    print("=" * 60)
    
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # 获取一个批次
    batch = next(iter(train_loader))
    
    print(f"\n{'Batch Information':^60}")
    print("=" * 60)
    print(f"RGB shape:      {batch['rgb'].shape}")
    print(f"Position shape: {batch['position'].shape}")
    print(f"Rotation shape: {batch['rotation'].shape}")
    
    print(f"\n{'Sample Data':^60}")
    print("=" * 60)
    print(f"Frame ID:   {batch['frame_id'][0]}")
    print(f"Position:   {batch['position'][0].numpy()}")
    print(f"Rotation:   {batch['rotation'][0].numpy()}")
    
    # 检查数据范围
    print(f"\n{'Data Statistics':^60}")
    print("=" * 60)
    print(f"Position range:")
    print(f"  X: [{batch['position'][:, 0].min():.3f}, {batch['position'][:, 0].max():.3f}]")
    print(f"  Y: [{batch['position'][:, 1].min():.3f}, {batch['position'][:, 1].max():.3f}]")
    print(f"  Z: [{batch['position'][:, 2].min():.3f}, {batch['position'][:, 2].max():.3f}]")
    
    print(f"Rotation range:")
    print(f"  Roll:  [{batch['rotation'][:, 0].min():.3f}, {batch['rotation'][:, 0].max():.3f}]")
    print(f"  Pitch: [{batch['rotation'][:, 1].min():.3f}, {batch['rotation'][:, 1].max():.3f}]")
    print(f"  Yaw:   [{batch['rotation'][:, 2].min():.3f}, {batch['rotation'][:, 2].max():.3f}]")
    
    print("\n" + "=" * 60)
    print("✓ DataLoader test passed!")
    print("=" * 60)
