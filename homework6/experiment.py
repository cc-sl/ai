#!/usr/bin/env python3
"""
YOLOv11 目标检测实验全流程脚本
环境：Python 3.12, PyTorch 2.x, CUDA 11.8, 单GPU (NVIDIA MX450 2GB)
数据集：PASCAL VOC 2007，位于 ./VOCdevkit
预训练权重：./yolo11n.pt
"""

import os
import sys
import yaml
import random
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from ultralytics import YOLO

# -------------------------- 配置文件 --------------------------

from config import Config
config = Config()

# -------------------------- 工具函数 --------------------------
def setup_directories():
    """创建结果保存目录"""
    dirs = ['runs', 'outputs', 'plots']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def set_seed(seed=42):
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def check_gpu_memory():
    """检查 GPU 显存并返回可用显存（GB）"""
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU 显存总量: {total_mem:.2f} GB")
        return total_mem

def check_file_exists(filepath, description):
    """检查文件是否存在，若存在则返回 True，否则打印提示"""
    if os.path.exists(filepath):
        print(f"✓ {description} 已存在: {filepath}")
        return True
    else:
        print(f"✗ {description} 不存在: {filepath}")
        return False

# -------------------------- 数据集预处理 --------------------------
def convert_voc_to_yolo(voc_dir, yolo_dir, small_mode=True, num_samples=300):
    """
    将 VOC 格式标注转换为 YOLO 格式，并创建 dataset.yaml。
    同时为小样本模式从 trainval 中随机抽取指定数量的图像。
    voc_dir: VOCdevkit 的上级路径（如 ./VOCdevkit/VOC2007）
    yolo_dir: 输出 YOLO 格式数据集的根目录
    small_mode: 是否启用小样本抽取
    num_samples: 小样本训练集图像数量
    """
    if os.path.exists(yolo_dir) and os.path.exists(os.path.join(yolo_dir, "dataset.yaml")):
        print(f"YOLO 数据集已存在，跳过转换: {yolo_dir}")
        return yolo_dir

    voc_path = Path(voc_dir) / "VOC2007"
    if not voc_path.exists():
        raise FileNotFoundError(f"VOC 数据集未找到: {voc_path}")

    # 解析类别名称
    classes = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    class_to_id = {name: idx for idx, name in enumerate(classes)}

    # 读取官方划分
    def read_image_set(split):
        split_file = voc_path / "ImageSets" / "Main" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"找不到 {split}.txt，请检查数据集结构")
        with open(split_file, 'r') as f:
            return [line.strip() for line in f.readlines()]

    trainval_ids = read_image_set("trainval")
    test_ids = read_image_set("test")

    # 小样本：从 trainval 中随机抽取 num_samples 张图像，保持类别分布均衡
    if small_mode:
        print(f"正在进行小样本抽取，目标数量: {num_samples}")
        random.shuffle(trainval_ids)

        # 统计每张图像的类别分布（粗略统计）
        img_to_classes = {}
        for img_id in trainval_ids:
            anno_file = voc_path / "Annotations" / f"{img_id}.xml"
            if not anno_file.exists():
                continue
            from xml.etree import ElementTree as ET
            tree = ET.parse(anno_file)
            objs = tree.findall('object')
            cats = [obj.find('name').text for obj in objs]
            img_to_classes[img_id] = set(cats)

        # 按类别均衡采样（简单贪心）
        selected = []
        quota = {c: max(1, num_samples // len(classes)) for c in classes}
        remaining = set(trainval_ids)
        # 先确保每个类别至少有一个样本
        for c in classes:
            candidate = None
            for img_id in list(remaining):
                if c in img_to_classes.get(img_id, set()):
                    candidate = img_id
                    break
            if candidate:
                selected.append(candidate)
                remaining.remove(candidate)
                quota[c] -= 1

        # 再随机填充其余样本
        need = num_samples - len(selected)
        if need > 0:
            extra = random.sample(list(remaining), min(need, len(remaining)))
            selected.extend(extra)

        # 划分训练集与验证集（9:1）
        random.shuffle(selected)
        split_idx = int(0.9 * len(selected))
        train_ids = selected[:split_idx]
        val_ids = selected[split_idx:]
        print(f"小样本训练集: {len(train_ids)} 张, 验证集: {len(val_ids)} 张")
    else:
        # 大样本：使用全部 trainval 作为训练集，验证集使用部分数据（如 10%）
        train_ids, val_ids = trainval_ids[:-1000], trainval_ids[-1000:]  # 简单划分

    # 创建 YOLO 目录结构
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(Path(yolo_dir) / sub, exist_ok=True)

    # 拷贝图片并生成 YOLO 标签
    def process_split(ids, split):
        for img_id in ids:
            img_path = voc_path / "JPEGImages" / f"{img_id}.jpg"
            if not img_path.exists():
                continue
            # 复制图片
            dest_img = Path(yolo_dir) / "images" / split / f"{img_id}.jpg"
            if not dest_img.exists():
                shutil.copy(img_path, dest_img)

            # 解析 XML 标注并写入 YOLO 格式
            anno_file = voc_path / "Annotations" / f"{img_id}.xml"
            if not anno_file.exists():
                continue
            tree = ET.parse(anno_file)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            label_lines = []
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in class_to_id:
                    continue
                cls_id = class_to_id[cls]
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                # 归一化中心点格式 [cx, cy, w, h]
                cx = ((xmin + xmax) / 2) / width
                cy = ((ymin + ymax) / 2) / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            dest_label = Path(yolo_dir) / "labels" / split / f"{img_id}.txt"
            with open(dest_label, 'w') as f:
                f.write("\n".join(label_lines))

    process_split(train_ids, "train")
    process_split(val_ids, "val")

    # 创建 dataset.yaml
    data_yaml = {
        'path': str(Path(yolo_dir).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(classes)}
    }
    with open(Path(yolo_dir) / "dataset.yaml", 'w') as f:
        yaml.dump(data_yaml, f)

    print(f"数据集转换完成，保存至 {yolo_dir}")
    return str(Path(yolo_dir).absolute())

# -------------------------- 训练与评估 --------------------------
import os
from ultralytics import YOLO  # 确保导入依赖


def train_model(data_yaml, weights, run_name, hyperparams, resume=False, retry_with_lower_lr=True):
    """
    通用训练函数。返回 (model, save_dir)。
    """
    try:
        model = YOLO(weights) if weights else YOLO('yolo11n.pt')

        results = model.train(
            data=data_yaml,
            epochs=hyperparams['epochs'],
            batch=hyperparams['batch'],
            imgsz=hyperparams['imgsz'],
            lr0=hyperparams['lr0'],
            optimizer=hyperparams['optimizer'],
            weight_decay=hyperparams['weight_decay'],
            momentum=hyperparams['momentum'],
            warmup_epochs=hyperparams['warmup_epochs'],
            amp=hyperparams['amp'],
            workers=hyperparams['workers'],
            resume=resume,
            project='runs/train',          # 保持原设置
            name=run_name,
            exist_ok=True,
            plots=False,                    #  关闭验证时绘图，避免 OverflowError
        )

        #  使用 ultralytics 实际保存的目录，而不是自己拼接
        save_dir = Path(results.save_dir)
        last_pt = save_dir / "weights" / "last.pt"
        best_pt = save_dir / "weights" / "best.pt"

        if not last_pt.exists() and not best_pt.exists():
            raise FileNotFoundError(
                f"训练未生成检查点。\n"
                f"  期望路径: {last_pt}\n"
                f"  实际目录内容: {list((save_dir / 'weights').glob('*')) if (save_dir / 'weights').exists() else 'weights目录不存在'}"
            )

        print(f"✓ 训练完成，模型保存至: {save_dir}")
        return model

    except FileNotFoundError as e:
        print(f"训练失败: {e}")
        if retry_with_lower_lr and hyperparams.get('lr0', 0.01) > 1e-4:
            new_lr = hyperparams['lr0'] / 10
            print(f"自动降低学习率至 {new_lr} 并重新训练...")
            new_hyper = hyperparams.copy()
            new_hyper['lr0'] = new_lr
            #  重试时用新名称，避免覆盖
            return train_model(data_yaml, weights, run_name + "_lr" + str(new_lr),
                               new_hyper, resume=False, retry_with_lower_lr=False)
        else:
            raise RuntimeError("训练因异常终止，无法保存模型。请检查数据集标签是否正确、学习率是否过高。")

def evaluate_model(model, data_yaml, iou_thresholds=None, save_txt=True):
    """
    使用指定的 IoU 阈值列表评估模型，返回各阈值下的指标字典。
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    metrics = {}
    for iou in iou_thresholds:
        print(f"\n评估 IoU={iou} ...")
        results = model.val(
            data=data_yaml,
            iou=iou,
            save_json=False,
            save_txt=save_txt,
            project='runs/val',
            name=f'iou_{iou}',
            exist_ok=True
        )
        metrics[iou] = {
            'mAP50': results.box.map50,
            'mAP50_95': results.box.map,
            'precision': results.box.p[0] if hasattr(results.box, 'p') else None,
            'recall': results.box.r[0] if hasattr(results.box, 'r') else None
        }
        print(f"  mAP@0.5: {metrics[iou]['mAP50']:.4f}, mAP@0.5:0.95: {metrics[iou]['mAP50_95']:.4f}")

    return metrics

# -------------------------- 可视化 --------------------------
def visualize_predictions(model, image_paths, output_dir='outputs/visual', confidence=0.5):
    """
    对给定图像进行推理，绘制真实框（从标注文件读取）与预测框的对比图。
    """
    os.makedirs(output_dir, exist_ok=True)

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue
        h, w = img.shape[:2]

        # 推理预测
        results = model(img)
        pred_boxes = results[0].boxes

        # 绘制真实框：尝试从同名标注文件读取
        gt_boxes = []
        label_path = img_path.replace('JPEGImages', 'Annotations').replace('jpg', 'xml')
        if os.path.exists(label_path):
            import xml.etree.ElementTree as ET
            tree = ET.parse(label_path)
            root = tree.getroot()
            for obj in root.iter('object'):
                cls = obj.find('name').text
                bbox = obj.find('bndbox')
                xmin = int(float(bbox.find('xmin').text))
                ymin = int(float(bbox.find('ymin').text))
                xmax = int(float(bbox.find('xmax').text))
                ymax = int(float(bbox.find('ymax').text))
                gt_boxes.append((cls, xmin, ymin, xmax, ymax))

        # 绘制真实框（绿色）
        for cls, x1, y1, x2, y2 in gt_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"GT:{cls}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 绘制预测框（红色）
        if pred_boxes is not None and len(pred_boxes) > 0:
            for box in pred_boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf.item()
                cls_id = int(box.cls.item())
                label = f"{model.names[cls_id]}: {conf:.2f}"
                if conf < confidence:
                    continue
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(img, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 保存结果
        out_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, img)
        print(f"可视化图片保存至: {out_path}")

# -------------------------- 主实验流程 --------------------------
def main():
    setup_directories()
    set_seed(42)
    check_gpu_memory()

    # 用户选择实验规模
    print("\n请选择实验模式:")
    print("1. 小样本实验 (300张训练，符合 2GB 显存)")
    print("2. 大样本实验 (需要 >2GB 显存，可能 OOM)")
    choice = input("输入 1 或 2 (默认 1): ").strip() or '1'
    if choice == '2':
        small_mode = False
        cfg = config.LARGE_SAMPLE
        print("已选择大样本实验，请注意显存！")
    else:
        small_mode = True
        cfg = config.SMALL_SAMPLE
        print("已选择小样本实验。")

    # 数据预处理（如已存在则跳过）
    yolo_data_dir = f"datasets/VOC2007_yolo_{'small' if small_mode else 'large'}"
    if os.path.exists(yolo_data_dir):
        print(f"数据集目录已存在，跳过转换: {yolo_data_dir}")
    else:
        convert_voc_to_yolo(config.DEVKIT_PATH, yolo_data_dir,
                            small_mode=small_mode, num_samples=cfg.get('num_samples', 300))

    # 准备数据 yaml 路径
    data_yaml = os.path.join(yolo_data_dir, "dataset.yaml")

    # 检查是否存在训练检查点
    run_name = f"yolo11n_voc_{'small' if small_mode else 'large'}"
    last_pt = f"runs/train/{run_name}/weights/last.pt"
    resume_training = False
    if os.path.exists(last_pt):
        print(f"发现上次训练检查点: {last_pt}")
        ans = input("是否继续上次的训练？(y/n, 默认 y): ").strip().lower() or 'y'
        resume_training = ans == 'y'

    # 基础训练（YOLOv11n）
    if resume_training:
        model = YOLO(last_pt)   # 加载上次最好的模型继续训练
    else:
        model = YOLO(config.WEIGHTS_PATH)

    # 训练（如果不需要训练可注释）
    print("\n开始训练...")
    model = train_model(data_yaml, config.WEIGHTS_PATH, run_name, cfg, resume=resume_training)

    # 基础评估（IoU=0.5 和 0.5:0.95）
    print("\n基础评估 (IoU=0.5)...")
    base_metrics = evaluate_model(model, data_yaml, iou_thresholds=[0.5, 0.95])

    # ---- 实验 1：不同 IoU 阈值对比评估 ----
    print("\n开始 IoU 阈值对比实验...")
    iou_metrics = evaluate_model(model, data_yaml, iou_thresholds=config.IOU_THRESHOLDS)
    # 保存结果到文件
    with open('outputs/iou_threshold_comparison.txt', 'w') as f:
        for iou, met in iou_metrics.items():
            f.write(f"IoU={iou}: mAP@0.5={met['mAP50']:.4f}, mAP@0.5:0.95={met['mAP50_95']:.4f}\n")
    print("IoU 对比结果已保存至 outputs/iou_threshold_comparison.txt")

    # ---- 实验 2：数据增强策略对比 ----
    print("\n开始数据增强对比实验...")
    for aug_name, aug_params in config.AUGMENT_SETTINGS.items():
        aug_cfg = cfg.copy()
        aug_cfg.update(aug_params)
        aug_run = f"{run_name}_aug_{aug_name}"
        print(f"训练增强策略: {aug_name}")
        aug_model = train_model(data_yaml, config.WEIGHTS_PATH, aug_run, aug_cfg, resume=False)
        aug_metrics = evaluate_model(aug_model, data_yaml, iou_thresholds=[0.5])
        print(f"  增强 {aug_name}  mAP@0.5: {aug_metrics[0.5]['mAP50']:.4f}")

    # ---- 实验 3：模型对比（可选，需下载对应权重） ----
    print("\n开始模型对比实验（若预训练权重不存在则跳过）...")
    for model_name, weight_path in config.MODEL_PATHS.items():
        if not os.path.exists(weight_path):
            print(f"跳过 {model_name}，权重文件不存在: {weight_path}")
            continue
        print(f"训练 {model_name} ...")
        model_run = f"{model_name}_voc_{'small' if small_mode else 'large'}"
        m = train_model(data_yaml, weight_path, model_run, cfg, resume=False)
        m_met = evaluate_model(m, data_yaml, iou_thresholds=[0.5])
        print(f"  {model_name} mAP@0.5: {m_met[0.5]['mAP50']:.4f}")

    # ---- 可视化典型案例 ----
    print("\n生成可视化结果...")
    # 选取测试集中的几张图片
    test_images = []
    test_dir = os.path.join(config.DEVKIT_PATH, "VOC2007", "JPEGImages")
    test_ids_file = os.path.join(config.DEVKIT_PATH, "VOC2007", "ImageSets", "Main", "test.txt")
    if os.path.exists(test_ids_file):
        with open(test_ids_file, 'r') as f:
            test_ids = [line.strip() for line in f.readlines()[:5]]  # 取前5张
        test_images = [os.path.join(test_dir, f"{img_id}.jpg") for img_id in test_ids]
    else:
        print("测试集文件缺失，无法生成可视化。")
    if test_images:
        visualize_predictions(model, test_images, output_dir='outputs/visual')

    print("\n所有实验完成！")

# -------------------------- 继续训练说明 --------------------------
def print_resume_instruction():
    print("""
如何继续训练：
1. 训练过程中会自动在 runs/train/<run_name>/weights/ 目录下保存 last.pt 和 best.pt。
2. 如果训练被中断（例如由于显存不足、手动停止），重新运行本脚本，会提示是否继续训练。
3. 选择 'y' 将加载 last.pt 并从中断的 epoch 继续（设置 resume=True）。
4. 如果希望从最佳检查点继续，可手动修改代码中的权重路径为 best.pt，并设置 resume=False（从该权重初始化训练）。
    """)

if __name__ == "__main__":
    print_resume_instruction()
    main()
