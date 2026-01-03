# config.py - 仅RGB版本
import os

# ============ 数据路径 ============
DATASET_ROOT = os.path.expanduser('~/pallet_dataset')

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
RESULT_DIR = os.path.join(OUTPUT_DIR, 'results')

# 创建目录
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ============ 数据集配置 ============
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# RGB 图像的归一化参数（ImageNet 统计）
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]

# ============ 模型配置 ============
MODEL_NAME = 'PalletPoseEstimator_EfficientNetB3_RGB_Euler'
BACKBONE = 'efficientnet-b3'
PRETRAINED = True
FEATURE_DIM = 1536  # EfficientNet-B3 特征维度

# ============ 训练超参数（3090 GPU优化）============
BATCH_SIZE = 64                  # 仅RGB，可以开到64甚至128
LEARNING_RATE = 1e-3             
WEIGHT_DECAY = 1e-4              
NUM_EPOCHS = 500                 
WARMUP_EPOCHS = 5                

# 损失函数权重
POSITION_LOSS_WEIGHT = 1.0
ROTATION_LOSS_WEIGHT = 1.0

# ============ 评估和保存 ============
EVAL_INTERVAL = 1                # 每个epoch都评估
SAVE_INTERVAL = 5               
PATIENCE = 100                    

# ============ 推理配置 ============
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
INFER_BATCH_SIZE = 1

# ============ 数据增强 ============
USE_AUGMENTATION = True
AUGMENT_RGB_BRIGHTNESS = 0.2
AUGMENT_RGB_CONTRAST = 0.2
AUGMENT_ROTATION_RANGE = 10


# ============ 学习率调度器配置 ============
LR_SCHEDULER = 'cosine'  # 'cosine', 'exponential', 'step', 'multistep'

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

