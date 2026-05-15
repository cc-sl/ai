import os
import shutil
import random
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

# ================= 1. 全局配置与超参数 =================
# 路径配置
VOC_ROOT = r"VOCdevkit\VOC2007"
YOLO_DATASET_ROOT = r"VOC_YOLO"
EXPERIMENT_ROOT = r"Experiments"

# 硬件与基础超参数 (针对2GB显存优化)
IMGSZ = 320          # 降低分辨率以适应2G显存 (原640会OOM)
BATCH_SIZE = 4       # 小显存需设置极小batch
DEVICE = 0 if torch.cuda.is_available() else 'cpu'   # 自动检测GPU
AMP = True           # 开启混合精度训练，必须开启以防OOM

# 实验超参数
EPOCHS_BASE = 50     # 基础训练轮数
LR_BASE = 0.01       # 初始学习率

# VOC的20个类别
VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
               'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# ================= 2. 数据预处理：VOC转YOLO格式 =================
def convert_voc_to_yolo(voc_root, yolo_root):
    """将VOC格式转换为YOLO格式，并生成对应的txt标签文件"""
    print("开始转换VOC到YOLO格式...")
    dirs = {
        'trainval': os.path.join(voc_root, 'ImageSets', 'Main', 'trainval.txt'),
        'test': os.path.join(voc_root, 'ImageSets', 'Main', 'test.txt')
    }

    for phase, txt_file in dirs.items():
        if not os.path.exists(txt_file):
            print(f"找不到 {txt_file}，请检查VOC数据集路径！")
            return

        with open(txt_file, 'r') as f:
            image_ids = [line.strip() for line in f.readlines()]

        # 创建YOLO目录结构
        img_out_dir = os.path.join(yolo_root, 'images', phase)
        label_out_dir = os.path.join(yolo_root, 'labels', phase)
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(label_out_dir, exist_ok=True)

        for img_id in image_ids:
            # 读取XML
            xml_path = os.path.join(voc_root, 'Annotations', f'{img_id}.xml')
            if not os.path.exists(xml_path):
                continue
            tree = ET.parse(xml_path)
            root = tree.getroot()

            img_w = int(root.find('size/width').text)
            img_h = int(root.find('size/height').text)

            yolo_labels = []
            for obj in root.iter('object'):
                cls_name = obj.find('name').text
                if cls_name not in VOC_CLASSES:
                    continue
                cls_id = VOC_CLASSES.index(cls_name)

                xmlbox = obj.find('bndbox')
                # 获取边界框坐标并转换为YOLO格式
                x_min = float(xmlbox.find('xmin').text)
                y_min = float(xmlbox.find('ymin').text)
                x_max = float(xmlbox.find('xmax').text)
                y_max = float(xmlbox.find('ymax').text)

                # 计算中心点与宽高，并归一化
                x_center = (x_min + x_max) / 2.0 / img_w
                y_center = (y_min + y_max) / 2.0 / img_h
                w = (x_max - x_min) / img_w
                h = (y_max - y_min) / img_h

                yolo_labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            # 写入标签文件
            label_path = os.path.join(label_out_dir, f'{img_id}.txt')
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_labels))

            # 复制图像文件
            src_img = os.path.join(voc_root, 'JPEGImages', f'{img_id}.jpg')
            dst_img = os.path.join(img_out_dir, f'{img_id}.jpg')
            if os.path.exists(src_img) and not os.path.exists(dst_img):
                shutil.copy(src_img, dst_img)

    print("VOC到YOLO格式转换完成！")

# ================= 3. 数据集划分：小样本 vs 大样本 =================
def create_few_shot_dataset(yolo_root, few_shot_root, num_samples=400):
    """
    创建小样本数据集（保证类别相对均衡的近似分层抽样）
    """
    print(f"开始生成 {num_samples} 张小样本数据集...")
    src_img_dir = os.path.join(yolo_root, 'images', 'trainval')
    src_lbl_dir = os.path.join(yolo_root, 'labels', 'trainval')

    dst_img_dir = os.path.join(few_shot_root, 'images', 'trainval')
    dst_lbl_dir = os.path.join(few_shot_root, 'labels', 'trainval')
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    # 🔧 修复1: 只获取 .jpg 图片文件，排除其他非图片文件
    all_imgs = sorted([f for f in os.listdir(src_img_dir) if f.endswith('.jpg')])

    # 统计每张图包含的类别
    img_classes = {}
    for img_name in all_imgs:
        lbl_path = os.path.join(src_lbl_dir, img_name.replace('.jpg', '.txt'))
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            classes_in_img = set(line.split()[0] for line in lines if line.strip())
            # 🔧 修复2: 忽略空标签的图片
            if classes_in_img:
                img_classes[img_name] = classes_in_img

    # 尝试按类别均衡抽取
    selected_imgs = set()
    class_count = {cls: 0 for cls in VOC_CLASSES}

    # 打乱图片顺序以增加随机性
    random.shuffle(all_imgs)

    for img_name in all_imgs:
        # 🔧 修复3: 跳过没有标签或标签为空的图片
        if img_name not in img_classes:
            continue
        if len(selected_imgs) >= num_samples:
            break
        # 获取当前图片中最稀缺的类别的计数
        min_cls_count = min(class_count[cls] for cls in img_classes[img_name])
        # 偏好选取包含稀缺类别的图片
        if min_cls_count < num_samples / 20 or len(selected_imgs) < num_samples * 0.8:
            selected_imgs.add(img_name)
            for cls in img_classes[img_name]:
                class_count[cls] += 1

    # 复制文件
    for img_name in selected_imgs:
        shutil.copy(os.path.join(src_img_dir, img_name), os.path.join(dst_img_dir, img_name))
        lbl_name = img_name.replace('.jpg', '.txt')
        lbl_src = os.path.join(src_lbl_dir, lbl_name)
        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, os.path.join(dst_lbl_dir, lbl_name))

    # 复制测试集(测试集大小样本实验共用)
    test_img_dst = os.path.join(few_shot_root, 'images', 'test')
    test_lbl_dst = os.path.join(few_shot_root, 'labels', 'test')
    if not os.path.exists(test_img_dst):
        # 🔧 修复4: 使用 dirs_exist_ok=True 避免中断后重新运行报错
        shutil.copytree(
            os.path.join(yolo_root, 'images', 'test'),
            test_img_dst,
            dirs_exist_ok=True
        )
        shutil.copytree(
            os.path.join(yolo_root, 'labels', 'test'),
            test_lbl_dst,
            dirs_exist_ok=True
        )

    print(f"小样本数据集生成完成，共抽取 {len(selected_imgs)} 张图像。")
    return len(selected_imgs)

def create_yaml(yolo_root, dataset_name):
    """生成Ultralytics需要的data.yaml配置文件"""
    yaml_path = os.path.join(yolo_root, 'data.yaml')
    data = {
        'path': os.path.abspath(yolo_root),
        'train': 'images/trainval',
        'val': 'images/test',
        'test': 'images/test',
        'names': {i: cls for i, cls in enumerate(VOC_CLASSES)}
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    return yaml_path

# ================= 4. 模型训练 =================
def train_model(yaml_path, exp_name, epochs=EPOCHS_BASE, batch=BATCH_SIZE, lr=LR_BASE):
    """使用YOLOv11n进行训练"""
    print(f"\n==== 开始训练实验: {exp_name} ====")
    # 加载预训练模型，加速收敛并适应小样本
    model = YOLO('yolo11n.pt')

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=IMGSZ,
        batch=batch,
        lr0=lr,
        device=DEVICE,
        amp=AMP,           # 混合精度
        cache=False,       # 2G显存不要开cache，会OOM
        project=EXPERIMENT_ROOT,
        name=exp_name,
        exist_ok=True
    )
    return os.path.join(EXPERIMENT_ROOT, exp_name, 'weights', 'best.pt')

# ================= 5. 模型评估：IoU阈值对比 =================
def evaluate_iou_thresholds(model_path, yaml_path, exp_name):
    """在不同IoU阈值下评估模型"""
    print(f"\n==== 评估 IoU 阈值影响: {exp_name} ====")
    model = YOLO(model_path)

    iou_thresholds = [0.3, 0.5, 0.7]
    eval_results = {}

    for iou in iou_thresholds:
        print(f"--- 评估 IoU={iou} ---")
        metrics = model.val(
            data=yaml_path,
            imgsz=IMGSZ,
            batch=BATCH_SIZE,
            device=DEVICE,
            iou=iou,            # NMS的IoU阈值
            conf=0.001,         # 置信度阈值拉低以计算完整的P-R曲线
            save_json=False,
            project=EXPERIMENT_ROOT,
            name=f"{exp_name}_eval_iou{iou}",
            exist_ok=True
        )
        # 记录核心指标
        eval_results[iou] = {
            'Precision': metrics.box.mp,    # Mean Precision
            'Recall': metrics.box.mr,       # Mean Recall
            'mAP@0.5': metrics.box.map50,   # mAP@0.5
            'mAP@0.5:0.95': metrics.box.map  # mAP@0.5:0.95
        }

    # 打印汇总表格
    print("\n==== IoU阈值对比结果 ====")
    print(f"{'IoU':<10} {'Precision':<12} {'Recall':<12} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15}")
    for iou, res in eval_results.items():
        print(f"{iou:<10} {res['Precision']:<12.4f} {res['Recall']:<12.4f} "
              f"{res['mAP@0.5']:<12.4f} {res['mAP@0.5:0.95']:<15.4f}")

    return eval_results

# ================= 6. 推理可视化：GT vs Prediction =================
def visualize_predictions(model_path, yaml_path, exp_name, num_images=5):
    """可视化真实框与预测框的对比"""
    print(f"\n==== 推理可视化: {exp_name} ====")
    model = YOLO(model_path)

    # 🔧 修复5: 简化路径推导，先查yaml所在目录，再fallback到VOC原始目录
    yaml_dir = os.path.dirname(yaml_path)
    test_img_dir = os.path.join(yaml_dir, 'images', 'test')
    test_lbl_dir = os.path.join(yaml_dir, 'labels', 'test')

    if not os.path.isdir(test_img_dir):
        test_img_dir = os.path.join(VOC_ROOT, 'JPEGImages')
        test_lbl_dir = os.path.join(YOLO_DATASET_ROOT, 'labels', 'test')

    if not os.path.isdir(test_img_dir):
        print(f"错误: 找不到测试图片目录! 尝试过: {test_img_dir}")
        return

    img_list = [f for f in os.listdir(test_img_dir) if f.endswith('.jpg')]
    if not img_list:
        print(f"警告: 测试图片目录为空: {test_img_dir}")
        return

    samples = random.sample(img_list, min(num_images, len(img_list)))

    out_dir = os.path.join(EXPERIMENT_ROOT, exp_name, "visualizations")
    os.makedirs(out_dir, exist_ok=True)

    for img_name in samples:
        img_path = os.path.join(test_img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图片 {img_path}，跳过")
            continue
        h, w = img.shape[:2]

        # 1. 绘制 GT (绿色)
        lbl_path = os.path.join(test_lbl_dir, img_name.replace('.jpg', '.txt'))
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                x_c, y_c, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                # 还原坐标
                x1 = int((x_c - bw / 2) * w)
                y1 = int((y_c - bh / 2) * h)
                x2 = int((x_c + bw / 2) * w)
                y2 = int((y_c + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for GT
                cv2.putText(img, f"GT:{VOC_CLASSES[cls_id]}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 2. 预测
        results = model.predict(img_path, imgsz=IMGSZ, conf=0.25, device=DEVICE, verbose=False)
        res = results[0]

        # 绘制 Prediction (红色)
        if res.boxes is not None:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for Pred
                cv2.putText(img, f"{VOC_CLASSES[cls_id]}:{conf:.2f}", (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 保存结果
        save_path = os.path.join(out_dir, f"vis_{img_name}")
        cv2.imwrite(save_path, img)
        print(f"已保存可视化结果: {save_path}")

# ================= 7. 主控流程 =================
if __name__ == "__main__":
    import time
    begin_time = time.time()
    print(f"实验开始时间: {time.ctime(begin_time)}")
    print("开始实验")

    # 0. 预备工作
    random.seed(42)  # 保证实验可复现
    os.makedirs(EXPERIMENT_ROOT, exist_ok=True)

    time1 = time.time()
    print(f"预备工作完成，耗时: {time1 - begin_time:.2f} 秒")

    # 1. 数据转换 (仅需运行一次，若已转换则注释掉)
    if not os.path.exists(YOLO_DATASET_ROOT):
        convert_voc_to_yolo(VOC_ROOT, YOLO_DATASET_ROOT)

    time2 = time.time()
    print(f"数据转换完成，耗时: {time2 - time1:.2f} 秒")

    # 2. 生成小样本与完整大样本的YAML配置
    # 大样本 (Full VOC2007 trainval - 约5000张)
    full_yaml = create_yaml(YOLO_DATASET_ROOT, "FullDataset")

    # 小样本 (约400张，均衡抽取)
    few_shot_root = os.path.join(os.path.dirname(YOLO_DATASET_ROOT), "VOC_YOLO_FewShot")
    if not os.path.exists(few_shot_root):
        create_few_shot_dataset(YOLO_DATASET_ROOT, few_shot_root, num_samples=400)
    few_yaml = create_yaml(few_shot_root, "FewShotDataset")

    time3 = time.time()
    print(f"数据集划分与YAML生成完成，耗时: {time3 - time2:.2f} 秒")

    # =========== 实验一：数据集对比实验 (小样本 vs 大样本) ===========
    # 训练小样本
    few_model_path = train_model(few_yaml, exp_name="FewShot_yolo11n", epochs=EPOCHS_BASE, lr=LR_BASE)
    time4 = time.time()
    print(f"小样本训练完成，耗时: {time4 - time3:.2f} 秒")
    # 评估IoU阈值影响
    evaluate_iou_thresholds(few_model_path, few_yaml, "FewShot_yolo11n")
    time5 = time.time()
    print(f"小样本评估完成，耗时: {time5 - time4:.2f} 秒")

    # 训练大样本
    # full_model_path = train_model(full_yaml, exp_name="FullShot_yolo11n", epochs=EPOCHS_BASE, lr=LR_BASE)

    # 评估IoU阈值影响 (使用大样本模型作为基准)
    # evaluate_iou_thresholds(full_model_path, full_yaml, "FullShot_yolo11n")

    # =========== 实验二：训练策略实验 (修改超参数) ===========
    # 降低学习率实验
    # train_model(full_yaml, exp_name="FullShot_LowLR", epochs=EPOCHS_BASE, lr=0.001)
    # 增加Epoch实验
    # train_model(full_yaml, exp_name="FullShot_MoreEpoch", epochs=100, lr=LR_BASE)

    # =========== 实验三：推理可视化 ===========
    # 选取大样本模型进行可视化
    # visualize_predictions(full_model_path, full_yaml, "FullShot_yolo11n", num_images=5)

    print("\n====== 所有实验运行完毕！请查看 Experiments 目录下结果 ======")
