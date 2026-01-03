#!/usr/bin/env python3
"""
深度图统计分析脚本

用于分析实际数据集中深度图的平均值、标准差、有效比例等统计信息
然后根据统计结果修改 config_rgbd.py 中的深度平均值和标准差
"""

import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse


def analyze_depth_directory(depth_dir, depth_min_mm=100, depth_max_mm=5000):
    """
    分析深度图像目录的统计信息
    
    Args:
        depth_dir: 深度图下网路径
        depth_min_mm: 最小有效深度 (mm)
        depth_max_mm: 最大有效深度 (mm)
        
    Returns:
        stats: 统计字典
    """
    depth_files = sorted(Path(depth_dir).glob('depth_*.png'))
    
    if len(depth_files) == 0:
        print(f"Error: No depth images found in {depth_dir}")
        return None
    
    print(f"Found {len(depth_files)} depth images")
    print("Analyzing...\n")
    
    all_valid_depths = []
    all_depths = []
    valid_pixel_ratios = []
    
    for depth_path in tqdm(depth_files, desc="Processing"):
        # 读取 16-bit 深度图
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        
        if depth is None:
            print(f"Warning: Failed to read {depth_path}")
            continue
        
        # uint16 深度值取整数部分（单位 mm）
        depth = depth.astype(np.uint16)
        
        # 水平：删除特殊值
        if depth.ndim > 2:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        
        # 统计所有数值
        all_depths.append(depth.flatten())
        
        # 统计有效深度
        valid_mask = (depth > 0) & (depth >= depth_min_mm) & (depth <= depth_max_mm)
        valid_depths = depth[valid_mask]
        
        if len(valid_depths) > 0:
            all_valid_depths.append(valid_depths)
        
        # 计算有效像素比例
        valid_ratio = valid_mask.sum() / depth.size
        valid_pixel_ratios.append(valid_ratio)
    
    # 合并所有推简
    all_valid_depths = np.concatenate(all_valid_depths)
    all_depths = np.concatenate(all_depths)
    
    # 计算统计信息
    stats = {
        'num_frames': len(depth_files),
        'total_pixels': all_depths.size,
        'valid_pixels': all_valid_depths.size,
        'valid_ratio_mean': np.mean(valid_pixel_ratios),
        'valid_ratio_min': np.min(valid_pixel_ratios),
        'valid_ratio_max': np.max(valid_pixel_ratios),
        
        # 有效深度统计
        'depth_min': all_valid_depths.min(),
        'depth_max': all_valid_depths.max(),
        'depth_mean': all_valid_depths.mean(),
        'depth_median': np.median(all_valid_depths),
        'depth_std': all_valid_depths.std(),
        'depth_q25': np.percentile(all_valid_depths, 25),
        'depth_q75': np.percentile(all_valid_depths, 75),
        
        # 所有深度统计（含 0 值）
        'all_depth_min': all_depths[all_depths > 0].min() if (all_depths > 0).any() else 0,
        'all_depth_max': all_depths[all_depths > 0].max() if (all_depths > 0).any() else 0,
        'all_depth_mean': all_depths[all_depths > 0].mean() if (all_depths > 0).any() else 0,
        'all_depth_std': all_depths[all_depths > 0].std() if (all_depths > 0).any() else 0,
        'zero_pixel_ratio': (all_depths == 0).sum() / all_depths.size,
    }
    
    return stats


def print_stats(stats):
    """打印统计信息"""
    print("\n" + "="*70)
    print("DEPTH IMAGE STATISTICS ANALYSIS".center(70))
    print("="*70)
    
    print(f"\n[数据集信息]")
    print(f"  Total frames: {stats['num_frames']}")
    print(f"  Total pixels: {stats['total_pixels']:,}")
    print(f"  Valid pixels: {stats['valid_pixels']:,} ({stats['valid_pixels']/stats['total_pixels']*100:.1f}%)")
    print(f"  Zero/Invalid pixels: {stats['total_pixels']-stats['valid_pixels']:,} ({(stats['total_pixels']-stats['valid_pixels'])/stats['total_pixels']*100:.1f}%)")
    
    print(f"\n[有效深度像素统计] (活跃像素)")
    print(f"  Min depth:    {stats['depth_min']:.1f} mm ({stats['depth_min']/1000:.2f} m)")
    print(f"  Max depth:    {stats['depth_max']:.1f} mm ({stats['depth_max']/1000:.2f} m)")
    print(f"  Mean depth:   {stats['depth_mean']:.1f} mm ({stats['depth_mean']/1000:.2f} m)")
    print(f"  Median depth: {stats['depth_median']:.1f} mm ({stats['depth_median']/1000:.2f} m)")
    print(f"  Std dev:      {stats['depth_std']:.1f} mm ({stats['depth_std']/1000:.2f} m)")
    print(f"  Q25:          {stats['depth_q25']:.1f} mm ({stats['depth_q25']/1000:.2f} m)")
    print(f"  Q75:          {stats['depth_q75']:.1f} mm ({stats['depth_q75']/1000:.2f} m)")
    
    print(f"\n[采集器输出深度统计] (含 0 值)")
    print(f"  Mean depth:   {stats['all_depth_mean']:.1f} mm ({stats['all_depth_mean']/1000:.2f} m)")
    print(f"  Std dev:      {stats['all_depth_std']:.1f} mm ({stats['all_depth_std']/1000:.2f} m)")
    print(f"  Zero pixels:  {stats['zero_pixel_ratio']*100:.1f}%")
    
    print(f"\n[有效像素比例]")
    print(f"  Mean:         {stats['valid_ratio_mean']*100:.1f}%")
    print(f"  Min:          {stats['valid_ratio_min']*100:.1f}%")
    print(f"  Max:          {stats['valid_ratio_max']*100:.1f}%")
    
    print(f"\n" + "="*70)
    print("[推荐配置修改] - 将以下值添加到 config_rgbd.py:")
    print("="*70)
    print(f"\nDEPTH_MEAN_MM = {stats['depth_mean']:.0f}      # 有效深度平均")
    print(f"DEPTH_STD_MM = {stats['depth_std']:.0f}       # 有效深度标准差")
    print(f"\n# 或者使用所有像素（含缺失值）：")
    print(f"# DEPTH_MEAN_MM = {stats['all_depth_mean']:.0f}")
    print(f"# DEPTH_STD_MM = {stats['all_depth_std']:.0f}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze depth image statistics')
    parser.add_argument('--dataset-dir', type=str, default=None,
                       help='Dataset directory (auto-detect latest if not specified)')
    parser.add_argument('--min-depth', type=float, default=100,
                       help='Minimum valid depth (mm)')
    parser.add_argument('--max-depth', type=float, default=5000,
                       help='Maximum valid depth (mm)')
    
    args = parser.parse_args()
    
    # 自动检测数据路径
    if args.dataset_dir is None:
        dataset_root = os.path.expanduser('~/pallet_dataset')
        if not os.path.exists(dataset_root):
            print(f"Error: Dataset root not found at {dataset_root}")
            return
        
        # 找最新的数据集目录
        subdirs = sorted([d for d in os.listdir(dataset_root) 
                         if os.path.isdir(os.path.join(dataset_root, d))])
        if not subdirs:
            print(f"Error: No dataset directories found in {dataset_root}")
            return
        
        latest_dir = os.path.join(dataset_root, subdirs[-1])
        depth_dir = os.path.join(latest_dir, 'depth')
    else:
        depth_dir = args.dataset_dir if args.dataset_dir.endswith('depth') else \
                    os.path.join(args.dataset_dir, 'depth')
    
    print(f"Analyzing depth images in: {depth_dir}\n")
    
    if not os.path.exists(depth_dir):
        print(f"Error: Depth directory not found: {depth_dir}")
        return
    
    # 分析
    stats = analyze_depth_directory(depth_dir, args.min_depth, args.max_depth)
    
    if stats is None:
        return
    
    # 打印统计信息
    print_stats(stats)


if __name__ == '__main__':
    main()
