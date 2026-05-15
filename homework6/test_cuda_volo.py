# encoding: utf-8
# CUDA 和 VOLO 训练功能测试脚本
# 测试环境是否能够正常进行 VOLO 训练

import torch
import sys
import time

print("=" * 60)
print("【1】环境基本信息")
print("=" * 60)
print(f"Python 版本: {sys.version}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

# ==================== CUDA 检测 ====================
print("\n" + "=" * 60)
print("【2】CUDA 详细检测")
print("=" * 60)

if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    # 测试基本张量运算
    print("\n【3】CUDA 张量运算测试")
    device = torch.device("cuda")
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    
    # 测试矩阵乘法
    start = time.time()
    z = torch.mm(x, y)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"  1000x1000 矩阵乘法耗时: {elapsed:.4f} 秒 ✅")
    
    # 测试更复杂的运算
    start = time.time()
    for _ in range(100):
        z = torch.mm(x, y)
        z = torch.relu(z)
        z = torch.sigmoid(z)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"  100 次迭代运算耗时: {elapsed:.4f} 秒 ✅")
    
    # 清理显存
    del x, y, z
    torch.cuda.empty_cache()
    
else:
    print("❌ CUDA 不可用，请检查 PyTorch 安装和驱动")
    print("尝试使用 CPU 模式...")
    device = torch.device("cpu")

# ==================== VOLO 模型测试 ====================
print("\n" + "=" * 60)
print("【4】VOLO 模型加载测试")
print("=" * 60)

try:
    import timm
    print(f"timm 版本: {timm.__version__}")
    
    # 列出可用的 VOLO 模型
    volo_models = [m for m in timm.list_models() if 'volo' in m]
    print(f"可用的 VOLO 模型: {volo_models}")
    
    if volo_models:
        model_name = volo_models[0]  # 使用第一个 VOLO 模型
        print(f"\n加载模型: {model_name}")
        
        # 用小版本的模型测试（volo_d1_224 相对较小）
        small_volo = [m for m in volo_models if 'd1' in m]
        if small_volo:
            model_name = small_volo[0]
        
        model = timm.create_model(model_name, pretrained=False, num_classes=1000)
        model = model.to(device)
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,} ✅")
        
        # ==================== 前向传播测试 ====================
        print("\n" + "=" * 60)
        print("【5】VOLO 前向传播测试（CPU/GPU）")
        print("=" * 60)
        
        batch_size = 8
        dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
        
        model.eval()
        with torch.no_grad():
            start = time.time()
            output = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - start
        
        print(f"  输入形状: {dummy_input.shape}")
        print(f"  输出形状: {output.shape}")
        print(f"  前向传播耗时: {elapsed:.4f} 秒 ✅")
        
        # ==================== 训练测试 ====================
        print("\n" + "=" * 60)
        print("【6】VOLO 短时训练测试")
        print("=" * 60)
        
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # 生成随机标签
        dummy_labels = torch.randint(0, 1000, (batch_size,)).to(device)
        
        num_steps = 5
        for step in range(num_steps):
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = loss_fn(output, dummy_labels)
            loss.backward()
            optimizer.step()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            current_memory = torch.cuda.memory_allocated(device) / 1024**3 if torch.cuda.is_available() else 0
            print(f"  Step [{step+1}/{num_steps}] Loss: {loss.item():.4f} | "
                  f"显存占用: {current_memory:.2f} GB")
        
        print("\n✅ VOLO 训练测试通过！CUDA 和 VOLO 均能正常工作。")
        
    else:
        print("❌ 未找到 VOLO 模型")
        
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请安装依赖: pip install timm torch")
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("【7】最终结论")
print("=" * 60)
if torch.cuda.is_available():
    print("✅ CUDA 可用，GPU 可以正常使用")
else:
    print("❌ CUDA 不可用，训练将使用 CPU（速度极慢）")
if 'model' in locals():
    print("✅ VOLO 模型加载和训练测试完成")
else:
    print("❌ VOLO 模型测试未通过")
