# config_rgbd.py - RGB-D 双模态数据配置
# 专门适配 ROS2 数据采集器输出的数据格式
import os

# ============ 数据路径 ============
DATASET_ROOT = os.path.expanduser('~/pallet_dataset')

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints_rgbd')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models_rgbd')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs_rgbd')
RESULT_DIR = os.path.join(OUTPUT_DIR, 'results_rgbd')

# 创建目录
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ============ 数据集配置 ============
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# ============ 深度图像处理配置 ============
# *** 重要：根据您的 ROS2 采集器，深度图已保存为 16-bit PNG，单位 mm ***
# 采集器代码中：depth_mm = (depth_img * 1000.0).astype(np.uint16)
# 所以深度值范围通常是 0-65535（mm），对应 0-65.535 m

# 深度图像格式
DEPTH_FORMAT = 'uint16'        # 16-bit 无符号整数
DEPTH_UNIT = 'mm'             # 单位：毫米
DEPTH_SCALE = 1.0             # 缩放因子（uint16已是mm，无需缩放）

# 深度图有效范围（毫米）
DEPTH_MIN_MM = 100.0           # 最小有效深度（100mm = 10cm）
DEPTH_MAX_MM = 5000.0          # 最大有效深度（5000mm = 5m）

# 深度图后处理
FILL_MISSING_DEPTH = True      # 是否填充缺失值（0值像素）
FILL_METHOD = 'nearest'        # 填充方法：'nearest', 'mean', 'median', 'inpaint'

# 深度图质量检查
MIN_VALID_DEPTH_RATIO = 0.5    # 有效像素最少占比（50%）

# 深度图统计参数（需根据实际数据集调整）
# 这些参数用于归一化深度图到 [-1, 1] 范围
# 建议：先运行数据采集，然后用 analyze_depth_stats.py 计算实际统计值
DEPTH_MEAN_MM = 2000.0         # 深度平均值（毫米） - 建议先检测
DEPTH_STD_MM = 1000.0          # 深度标准差（毫米） - 建议先检测

# RGB 图像的归一化参数（ImageNet 统计）
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]

# ============ 模型配置 ============
MODEL_NAME = 'PalletPoseEstimator_EfficientNetB3_RGBD_Euler'
BACKBONE = 'efficientnet-b3'
PRETRAINED = True
FEATURE_DIM = 1536  # EfficientNet-B3 特征维度

# RGB-D融合方式
FUSION_STRATEGY = 'late'       # 'early' (输入处), 'late' (特征处), 'conv' (卷积融合)
                               # 推荐 'late' 因为RGB和Depth的特征差异大
DEPTH_CHANNELS = 1             # 深度图通道数

# ============ 训练超参数（3090 GPU优化）============
BATCH_SIZE = 32                # RGB-D双流，改为32（内存占用增加）
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 500
WARMUP_EPOCHS = 5

# 损失函数权重
POSITION_LOSS_WEIGHT = 1.0
ROTATION_LOSS_WEIGHT = 1.0

# ============ 评估和保存 ============
EVAL_INTERVAL = 1              # 每个epoch都评估
SAVE_INTERVAL = 5              # 每5个epoch保存一次
PATIENCE = 100                 # 早停耐心值

# ============ 推理配置 ============
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
INFER_BATCH_SIZE = 1

# ============ 数据增强 ============
USE_AUGMENTATION = True

# RGB 增强
AUGMENT_RGB_BRIGHTNESS = 0.2   # ±20% 亮度
AUGMENT_RGB_CONTRAST = 0.2     # ±20% 对比度

# 深度增强
AUGMENT_DEPTH_NOISE = 0.05     # 高斯噪声标准差（相对于depth_std）
AUGMENT_DEPTH_DROPOUT = 0.1    # 随机像素缺失比例（10%）
AUGMENT_DEPTH_BLUR = True      # 高斯模糊

# 空间增强（同时应用于RGB和Depth）
AUGMENT_ROTATION_RANGE = 10    # 旋转角度范围（度）
AUGMENT_FLIP_H = False         # 水平翻转
AUGMENT_FLIP_V = False         # 竖直翻转

# ============ 学习率调度器配置 ============
LR_SCHEDULER = 'cosine'        # 'cosine', 'exponential', 'step', 'multistep'

# Cosine Annealing
COSINE_T_MAX = NUM_EPOCHS
COSINE_ETA_MIN = 1e-6

# Exponential
EXPONENTIAL_GAMMA = 0.95

# Step
STEP_SIZE = 10
STEP_GAMMA = 0.5

# MultiStep
MULTISTEP_MILESTONES = [30, 60, 90]
MULTISTEP_GAMMA = 0.1

# ============ RGB-D特定配置 ============
# 深度编码器架构
DEPTH_ENCODER_TYPE = 'simple'  # 'simple', 'resnet', 'efficientnet'
                               # 推荐 'simple' 保持轻量级

# 特征融合权重
RGB_WEIGHT = 0.6               # RGB特征权重
DEPTH_WEIGHT = 0.4             # 深度特征权重

# ============ 数据采集兼容性配置 ============
# 与 ROS2 数据采集器的兼容性设置
COLLECTOR_DEPTH_UNIT = 'mm'    # 采集器输出深度单位：'mm' 或 'm'
COLLECTOR_OUTPUT_ROOT = os.path.expanduser('~/pallet_dataset')  # 采集器输出目录

# 文件名模式（与采集器一致）
RGB_FILENAME_PATTERN = 'rgb_{:06d}.png'
DEPTH_FILENAME_PATTERN = 'depth_{:06d}.png'
POSE_FILENAME = 'poses.txt'

# 深度图有效性检查
# 采集器中：depth_mm = (depth_img * 1000.0).astype(np.uint16)
# 所以 0 值表示缺失/无效深度
INVALID_DEPTH_VALUE = 0        # 缺失深度的标记值

print(f"""
{'='*60}
RGB-D Pose Estimation Configuration
{'='*60}
Dataset Root: {DATASET_ROOT}
Output Root: {OUTPUT_DIR}
Depth Format: {DEPTH_FORMAT} ({DEPTH_UNIT})
Depth Range: {DEPTH_MIN_MM:.0f}-{DEPTH_MAX_MM:.0f} {DEPTH_UNIT}
Depth Mean: {DEPTH_MEAN_MM:.0f} {DEPTH_UNIT}, Std: {DEPTH_STD_MM:.0f} {DEPTH_UNIT}
Fusion Strategy: {FUSION_STRATEGY}
Batch Size: {BATCH_SIZE}
Learning Rate: {LEARNING_RATE}
{'='*60}
""")
