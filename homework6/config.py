from pathlib import Path
class Config:
    """
    超参数配置文件，包含小样本和大样本两套参数。
    用户可根据实际显存修改 batch size 和图像尺寸。
    """
    # 基础路径
    DEVKIT_PATH = Path("./VOCdevkit")
    WEIGHTS_PATH = Path("./yolo11n.pt")
    # 小样本参数（最大 2GB 显存）
    # SMALL_SAMPLE = {
    #     'num_samples': 300,          # 抽取 300 张作为训练集
    #     'epochs': 50,
    #     'batch': 4,                  # MX450 2GB 建议 batch=4
    #     'imgsz': 640,
    #     #'lr0': 0.01,                 # 初始学习率
    #     'lr0': 0.001,
    #     'optimizer': 'SGD',
    #     'weight_decay': 0.0005,
    #     'momentum': 0.937,
    #     'warmup_epochs': 3,
    #     'amp': True,                 # 混合精度训练
    #     'workers': 4,
    #     'data_augment': 'default',   # 使用默认增强
    # }
    # SMALL_SAMPLE = {
    #     'num_samples': 300,
    #     'epochs': 50,
    #     'batch': 4,          # 可尝试保持
    #     'imgsz': 416,        # 关键降低
    #     'lr0': 0.001,        # 降低学习率
    #     'optimizer': 'SGD',
    #     'weight_decay': 0.0005,
    #     'momentum': 0.937,
    #     'warmup_epochs': 3,
    #     'amp': True,
    #     'workers': 1,        # 减少内存压力
    #     'data_augment': 'default',
    # }

    SMALL_SAMPLE = {
        'num_samples': 300,
        'epochs': 30,           #  50 → 30
        'batch': 8,             #  4 → 8（配合更小 imgsz）
        'imgsz': 320,           #  416 → 320
        'lr0': 0.001,
        'optimizer': 'AdamW',   #  SGD → AdamW，小数据集收敛更快
        'weight_decay': 0.0005,
        'momentum': 0.937,
        'warmup_epochs': 2,     #  3 → 2
        'amp': False,
        'workers': 2,           #  1 → 2
        'data_augment': 'default',
    }

    # 大样本参数（此处假设另外有 6GB 显存可用，实际若显存不足会 OOM）
    LARGE_SAMPLE = {
        'epochs': 100,
        'batch': 8,                  # 6GB 显存可适当调大
        'imgsz': 640,
        'lr0': 0.01,
        'optimizer': 'SGD',
        'weight_decay': 0.0005,
        'momentum': 0.937,
        'warmup_epochs': 3,
        'amp': True,
        'workers': 4,
        'data_augment': 'default',
    }
    # 数据增强对比实验参数
    AUGMENT_SETTINGS = {
        'none': {
            'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0,
            'degrees': 0.0, 'translate': 0.0, 'scale': 0.0,
            'shear': 0.0, 'flipud': 0.0, 'fliplr': 0.0,
            'mosaic': 0.0, 'erasing': 0.0
        },
        'light': {
            'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
            'degrees': 0.0, 'translate': 0.1, 'scale': 0.5,
            'shear': 0.0, 'flipud': 0.0, 'fliplr': 0.5,
            'mosaic': 0.5, 'erasing': 0.0
        },
        'heavy': {
            'hsv_h': 0.03, 'hsv_s': 1.0, 'hsv_v': 0.8,
            'degrees': 30.0, 'translate': 0.3, 'scale': 0.9,
            'shear': 10.0, 'flipud': 0.5, 'fliplr': 0.5,
            'mosaic': 1.0, 'erasing': 0.4
        }
    }
    # IoU 阈值列表（用于评估对比）
    IOU_THRESHOLDS = [0.3, 0.5, 0.7]
    # 模型对比实验（需提前下载对应的预训练权重）
    MODEL_PATHS = {
        'yolov11n': './yolo11n.pt',
        'yolov11s': './yolo11s.pt',   # 用户需自行下载
        'yolov11m': './yolo11m.pt'
    }
