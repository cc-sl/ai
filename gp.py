import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV
df = pd.read_csv("runs\detect\\runs\\train\yolo11n_voc_small_aug_none\\results.csv")

# 清理列名（去空格）
df.columns = df.columns.str.strip()

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.figure(figsize=(15, 10))

# ========== 子图 1：精度曲线 ==========
plt.subplot(2,2,1)
plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
plt.xlabel('Epoch')
plt.ylabel('指标值')
plt.title('精度指标变化曲线')
plt.legend()
plt.grid(alpha=0.3)

# ========== 子图 2：训练损失 ==========
plt.subplot(2,2,2)
plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss')
plt.plot(df['epoch'], df['train/cls_loss'], label='Cls Loss')
plt.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss')
plt.xlabel('Epoch')
plt.ylabel('训练损失')
plt.title('训练集损失变化曲线')
plt.legend()
plt.grid(alpha=0.3)

# ========== 子图 3：验证损失 ==========
plt.subplot(2,2,3)
plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
plt.plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss')
plt.plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss')
plt.xlabel('Epoch')
plt.ylabel('验证损失')
plt.title('验证集损失变化曲线')
plt.legend()
plt.grid(alpha=0.3)

# ========== 子图 4：学习率 ==========
plt.subplot(2,2,4)
plt.plot(df['epoch'], df['lr/pg0'], label='LR pg0')
plt.plot(df['epoch'], df['lr/pg1'], label='LR pg1')
plt.plot(df['epoch'], df['lr/pg2'], label='LR pg2')
plt.xlabel('Epoch')
plt.ylabel('学习率')
plt.title('学习率下降曲线')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('训练曲线.png', dpi=300)
plt.show()