# -*- coding: utf-8 -*-
"""
实验名称：YOLOv11 在 PASCAL VOC 2007 上的目标检测实验
实验内容：
  1. 数据集预处理（VOC → YOLO 格式），trainval 与 test 分目录存放
  2. 小样本均衡子集划分（200~500 张）
  3. 大样本（完整 trainval）训练对比
  4. 不同 IoU 阈值（0.3, 0.5, 0.7）评估 mAP、Precision、Recall
  5. 预测结果可视化
硬件限制：NVIDIA MX450 2GB，需设置较小的 batch size 和图像尺寸。
"""

import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

# ==================== 配置参数 ====================
# 数据集路径
VOC_ROOT = Path("VOCdevkit/VOC2007")          # VOC2007 根目录

# YOLO 格式输出目录（trainval 和 test 分开存放）
YOLO_DIR = Path("VOC_YOLO")
TRAINVAL_IMAGES = YOLO_DIR / "trainval" / "images"
TRAINVAL_LABELS = YOLO_DIR / "trainval" / "labels"
TEST_IMAGES     = YOLO_DIR / "test" / "images"
TEST_LABELS     = YOLO_DIR / "test" / "labels"

# 实验输出目录
EXPERIMENT_DIR = Path("experiments")
SMALL_SAMPLE_DIR = EXPERIMENT_DIR / "small_sample"
LARGE_SAMPLE_DIR = EXPERIMENT_DIR / "large_sample"

# 小样本设置
SMALL_SAMPLE_SIZE = 400         # 抽取图像数量（200~500）
SMALL_SAMPLE_SEED = 42

# 训练超参数（适应低显存 MX450 2GB）
BATCH_SIZE = 4                  # 显存有限，建议 ≤ 4
IMAGE_SIZE = 320                # 降低分辨率以节省显存
EPOCHS = 50
LR0 = 0.01                      # 初始学习率
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

# IoU 阈值列表（用于对比评估）
IOU_THRESHOLDS = [0.3, 0.5, 0.7]

# VOC 类别（20类）
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# 确保输出目录存在
for d in [TRAINVAL_IMAGES, TRAINVAL_LABELS, TEST_IMAGES, TEST_LABELS,
          SMALL_SAMPLE_DIR, LARGE_SAMPLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ==================== 工具函数 ====================
def parse_voc_xml(xml_path: Path) -> tuple:
    """
    解析 VOC XML 标注文件。
    返回:
        objects: [(class_name, xmin, ymin, xmax, ymax), ...]
        width, height: 图像尺寸
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in VOC_CLASSES:
            continue
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        objects.append((name, xmin, ymin, xmax, ymax))
    return objects, width, height


def convert_voc_subset(voc_dir: Path, dst_img_dir: Path, dst_lbl_dir: Path,
                       subset: str = "trainval") -> list:
    """
    转换 VOC 某个子集（trainval / test）为 YOLO 格式，
    图像和标签分别存入 dst_img_dir / dst_lbl_dir。
    返回该子集所有图像的绝对路径列表。
    """
    img_dir = voc_dir / "JPEGImages"
    anno_dir = voc_dir / "Annotations"
    set_file = voc_dir / "ImageSets" / "Main" / f"{subset}.txt"

    if not set_file.exists():
        raise FileNotFoundError(f"找不到 {set_file}，请检查 VOC 数据集是否完整。")

    with open(set_file, 'r') as f:
        image_ids = [line.strip() for line in f.readlines()]

    image_paths = []
    for img_id in tqdm(image_ids, desc=f"转换 {subset} 数据"):
        xml_path = anno_dir / f"{img_id}.xml"
        if not xml_path.exists():
            print(f"警告：标注文件 {xml_path} 不存在，跳过")
            continue

        objects, img_w, img_h = parse_voc_xml(xml_path)

        # 复制图像
        src_img = img_dir / f"{img_id}.jpg"
        if not src_img.exists():
            print(f"警告：图像 {src_img} 不存在，跳过")
            continue
        dst_img = dst_img_dir / f"{img_id}.jpg"
        shutil.copy2(src_img, dst_img)
        image_paths.append(str(dst_img.resolve()))

        # 生成 YOLO 标签（归一化中心点 + 宽高）
        label_lines = []
        for cls_name, xmin, ymin, xmax, ymax in objects:
            cls_id = VOC_CLASSES.index(cls_name)
            x_center = ((xmin + xmax) / 2.0) / img_w
            y_center = ((ymin + ymax) / 2.0) / img_h
            bw = (xmax - xmin) / img_w
            bh = (ymax - ymin) / img_h
            label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

        lbl_file = dst_lbl_dir / f"{img_id}.txt"
        with open(lbl_file, 'w') as f:
            f.write('\n'.join(label_lines))

    return image_paths


def sample_balanced_subset(image_paths: list, labels_dir: Path,
                           n: int, seed: int) -> list:
    """
    从图像列表中均衡抽取 n 张，使类别分布尽量均衡。
    策略：统计每张图的类别稀有度，优先选取包含稀有类别的图像。
    """
    # 统计每张图的类别
    img2classes = defaultdict(set)
    all_classes = []
    for img_path in image_paths:
        img_id = Path(img_path).stem
        lbl_path = labels_dir / f"{img_id}.txt"
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                class_ids = [int(line.split()[0]) for line in f.readlines()]
            img2classes[img_id] = set(class_ids)
            all_classes.extend(class_ids)

    # 类别计数
    class_counter = Counter(all_classes)
    print(f"训练集各类别样本数: {dict(class_counter)}")

    # 计算每张图的"稀有度分数"：包含的类别越稀有，分数越高
    rarity_score = {}
    for img_id, cls_set in img2classes.items():
        score = sum(1.0 / max(class_counter.get(c, 1), 1) for c in cls_set)
        rarity_score[img_id] = score

    # 按稀有度降序选取（优先包含少样本类别）
    sorted_ids = sorted(rarity_score.keys(),
                        key=lambda x: rarity_score[x], reverse=True)

    random.seed(seed)
    selected = []
    selected_set = set()
    for img_id in sorted_ids:
        if len(selected) >= n:
            break
        if img_id not in selected_set:
            selected.append(img_id)
            selected_set.add(img_id)

    # 若不足，随机补充
    if len(selected) < n:
        remaining = [Path(p).stem for p in image_paths
                     if Path(p).stem not in selected_set]
        additional = random.sample(remaining, n - len(selected))
        selected.extend(additional)

    # 统计选中图像的类别分布
    selected_classes = []
    for img_id in selected:
        lbl_path = labels_dir / f"{img_id}.txt"
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                selected_classes.extend(
                    [int(line.split()[0]) for line in f.readlines()])
    print(f"小样本各类别分布: {dict(Counter(selected_classes))}")

    # 返回完整图像路径
    path_dict = {Path(p).stem: p for p in image_paths}
    return [path_dict[img_id] for img_id in selected]


def create_dataset_subset(image_paths: list, dest_img_dir: Path,
                          dest_lbl_dir: Path, src_lbl_dir: Path):
    """将选定的图像和对应标签复制到目标目录"""
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lbl_dir.mkdir(parents=True, exist_ok=True)
    for img_path in image_paths:
        img_id = Path(img_path).stem
        shutil.copy2(img_path, dest_img_dir / f"{img_id}.jpg")
        src_lbl = src_lbl_dir / f"{img_id}.txt"
        if src_lbl.exists():
            shutil.copy2(src_lbl, dest_lbl_dir / f"{img_id}.txt")
    print(f"已创建子集：{len(image_paths)} 张图像 -> {dest_img_dir}")


# ==================== 训练与评估函数 ====================
def train_yolo(data_yaml: str, project_dir: str, name: str,
               exist_ok: bool = True) -> Path:
    """
    使用 YOLOv11n 训练模型。
    返回最佳权重文件路径。
    """
    model = YOLO("yolo11n.pt")  # 加载预训练权重
    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        lr0=LR0,
        device=DEVICE,
        project=project_dir,
        name=name,
        exist_ok=exist_ok,
        verbose=True,
        amp=True if DEVICE != 'cpu' else False,  # 混合精度，节省显存
        workers=2,
    )
    best_path = Path(project_dir) / name / "weights" / "best.pt"
    return best_path


def evaluate_iou_thresholds(model_path: str, data_yaml: str,
                            iou_vals: list) -> dict:
    """
    在不同 IoU 阈值下评估模型。
    返回 {iou_value: {mAP@0.5, mAP@0.5:0.95, precision, recall}} 格式的字典。
    """
    results_dict = {}
    for iou in iou_vals:
        model = YOLO(model_path)
        metrics = model.val(
            data=data_yaml,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            iou=iou,
            conf=0.001,
            verbose=False,
            plots=False,
        )
        results_dict[iou] = {
            "mAP@0.5": float(metrics.box.map50),
            "mAP@0.5:0.95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
    return results_dict


def visualize_predictions(model_path: str, image_paths: list,
                          labels_dir: Path, output_dir: Path,
                          num_samples: int = 5):
    """
    对采样图像进行预测，并可视化真实框（GT，绿色）与预测框（红色）。
    """
    model = YOLO(model_path)
    selected = random.sample(image_paths, min(num_samples, len(image_paths)))
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 5 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for idx, img_path in enumerate(selected):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # --- 绘制真实框（绿色）---
        lbl_path = labels_dir / f"{Path(img_path).stem}.txt"
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id, cx, cy, bw, bh = map(float, parts)
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    box_w = int(bw * w)
                    box_h = int(bh * h)
                    rect = patches.Rectangle(
                        (x1, y1), box_w, box_h,
                        linewidth=2, edgecolor='lime', facecolor='none'
                    )
                    axes[idx].add_patch(rect)
                    axes[idx].text(
                        x1, max(y1 - 5, 10),
                        f"GT:{VOC_CLASSES[int(cls_id)]}",
                        color='lime', fontsize=7,
                        bbox=dict(facecolor='white', alpha=0.7, pad=0)
                    )

        # --- 模型预测 ---
        results = model.predict(
            img_path, imgsz=IMAGE_SIZE, conf=0.25,
            device=DEVICE, verbose=False
        )
        pred_boxes = results[0].boxes

        # --- 绘制预测框（红色）---
        if pred_boxes is not None and len(pred_boxes) > 0:
            for box in pred_boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                axes[idx].add_patch(rect)
                axes[idx].text(
                    x1, max(y2 + 12, 10),
                    f"PD:{VOC_CLASSES[cls_id]} {conf:.2f}",
                    color='red', fontsize=7,
                    bbox=dict(facecolor='white', alpha=0.7, pad=0)
                )

        axes[idx].imshow(img)
        axes[idx].set_title(f"Image: {Path(img_path).name}", fontsize=10)
        axes[idx].axis('off')

    plt.tight_layout()
    out_path = output_dir / "predictions_vs_gt.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"可视化结果已保存至 {out_path}")


def write_data_yaml(train_img_dir: Path, val_img_dir: Path,
                    output_path: Path):
    """
    写入 ultralytics 格式的 data.yaml。
    train/val 使用绝对路径，不再设置全局 'path' 字段（避免路径拼接问题）。
    """
    import yaml
    config = {
        # 不设置 path，train/val 直接使用绝对路径
        "train": str(train_img_dir.resolve()),
        "val": str(val_img_dir.resolve()),
        "test": str(val_img_dir.resolve()),  # test 复用 val（即 VOC test）
        "names": {i: name for i, name in enumerate(VOC_CLASSES)},
        "nc": len(VOC_CLASSES),
    }
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"YAML 配置已写入: {output_path}")
    print(f"  train: {config['train']}")
    print(f"  val:   {config['val']}")


# ==================== 主实验流程 ====================
def main():
    print("=" * 60)
    print("==> 步骤 1：数据集格式转换 (VOC → YOLO)")
    print("=" * 60)

    # 分别转换 trainval 和 test，存入不同目录
    trainval_paths = convert_voc_subset(
        VOC_ROOT, TRAINVAL_IMAGES, TRAINVAL_LABELS, subset="trainval"
    )
    test_paths = convert_voc_subset(
        VOC_ROOT, TEST_IMAGES, TEST_LABELS, subset="test"
    )
    print(f"训练集图像数: {len(trainval_paths)}")
    print(f"测试集图像数: {len(test_paths)}")

    # ==================== 小样本实验 ====================
    print("\n" + "=" * 60)
    print(f"==> 步骤 2：构建小样本均衡子集（目标 {SMALL_SAMPLE_SIZE} 张）")
    print("=" * 60)

    small_img_paths = sample_balanced_subset(
        trainval_paths, TRAINVAL_LABELS,
        SMALL_SAMPLE_SIZE, SMALL_SAMPLE_SEED
    )

    small_train_img = SMALL_SAMPLE_DIR / "train" / "images"
    small_train_lbl = SMALL_SAMPLE_DIR / "train" / "labels"
    create_dataset_subset(small_img_paths, small_train_img,
                          small_train_lbl, TRAINVAL_LABELS)

    # 写入小样本 YAML 配置
    small_yaml_path = EXPERIMENT_DIR / "small_data.yaml"
    write_data_yaml(small_train_img, TEST_IMAGES, small_yaml_path)

    print("\n" + "=" * 60)
    print("==> 步骤 3：训练小样本模型 (YOLOv11n)")
    print("=" * 60)
    small_model_path = train_yolo(
        str(small_yaml_path), str(EXPERIMENT_DIR), "small_sample"
    )
    print(f"小样本模型最佳权重: {small_model_path}")

    # ==================== 大样本实验 ====================
    print("\n" + "=" * 60)
    print("==> 步骤 4：构建大样本数据集（完整 trainval）")
    print("=" * 60)

    # 大样本直接使用转换好的 TRAINVAL_IMAGES / TRAINVAL_LABELS
    # 写入大样本 YAML 配置
    large_yaml_path = EXPERIMENT_DIR / "large_data.yaml"
    write_data_yaml(TRAINVAL_IMAGES, TEST_IMAGES, large_yaml_path)

    print("\n" + "=" * 60)
    print("==> 步骤 5：训练大样本模型 (YOLOv11n)")
    print("=" * 60)
    large_model_path = train_yolo(
        str(large_yaml_path), str(EXPERIMENT_DIR), "large_sample"
    )
    print(f"大样本模型最佳权重: {large_model_path}")

    # ==================== IoU 阈值评估 ====================
    print("\n" + "=" * 60)
    print("==> 步骤 6：不同 IoU 阈值评估")
    print("=" * 60)

    print("\n--- 小样本模型评估 ---")
    small_iou_metrics = evaluate_iou_thresholds(
        str(small_model_path), str(small_yaml_path), IOU_THRESHOLDS
    )

    print("\n--- 大样本模型评估 ---")
    large_iou_metrics = evaluate_iou_thresholds(
        str(large_model_path), str(large_yaml_path), IOU_THRESHOLDS
    )

    # 打印结果对比表格
    print("\n" + "=" * 70)
    print("评估结果对比：小样本 ({} 张) vs 大样本 ({} 张)".format(
        SMALL_SAMPLE_SIZE, len(trainval_paths)))
    print("=" * 70)
    header = (f"{'IoU':>6} | {'模型':<6} | {'mAP@0.5':>10} | "
              f"{'mAP@0.5:0.95':>13} | {'Precision':>10} | {'Recall':>10}")
    print(header)
    print("-" * 70)
    for iou in IOU_THRESHOLDS:
        s = small_iou_metrics[iou]
        l_ = large_iou_metrics[iou]
        print(f"{iou:>6.1f} | {'小样本':<6} | {s['mAP@0.5']:>10.4f} | "
              f"{s['mAP@0.5:0.95']:>13.4f} | {s['precision']:>10.4f} | "
              f"{s['recall']:>10.4f}")
        print(f"{'':>6} | {'大样本':<6} | {l_['mAP@0.5']:>10.4f} | "
              f"{l_['mAP@0.5:0.95']:>13.4f} | {l_['precision']:>10.4f} | "
              f"{l_['recall']:>10.4f}")
        print("-" * 70)

    # 保存评估结果到文本文件
    result_file = EXPERIMENT_DIR / "evaluation_results.txt"
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("评估结果对比\n")
        f.write(f"小样本: {SMALL_SAMPLE_SIZE} 张, 大样本: {len(trainval_paths)} 张\n\n")
        f.write(header + "\n")
        f.write("-" * 70 + "\n")
        for iou in IOU_THRESHOLDS:
            s = small_iou_metrics[iou]
            l_ = large_iou_metrics[iou]
            f.write(f"{iou:>6.1f} | {'小样本':<6} | {s['mAP@0.5']:>10.4f} | "
                    f"{s['mAP@0.5:0.95']:>13.4f} | {s['precision']:>10.4f} | "
                    f"{s['recall']:>10.4f}\n")
            f.write(f"{'':>6} | {'大样本':<6} | {l_['mAP@0.5']:>10.4f} | "
                    f"{l_['mAP@0.5:0.95']:>13.4f} | {l_['precision']:>10.4f} | "
                    f"{l_['recall']:>10.4f}\n")
            f.write("-" * 70 + "\n")
    print(f"\n评估结果已保存至 {result_file}")

    # ==================== 可视化 ====================
    print("\n" + "=" * 60)
    print("==> 步骤 7：预测可视化（大样本模型在测试集上的表现）")
    print("=" * 60)
    visualize_predictions(
        str(large_model_path), test_paths, TEST_LABELS,
        EXPERIMENT_DIR / "visualizations", num_samples=5
    )

    print("\n" + "=" * 60)
    print("实验全部完成！请查看 experiments 目录下的输出。")
    print("=" * 60)


if __name__ == "__main__":
    # 依赖检查
    try:
        import yaml
        import ultralytics
    except ImportError:
        print("请先安装所需依赖库:")
        print("  pip install ultralytics opencv-python matplotlib pyyaml tqdm")
        exit(1)

    main()
