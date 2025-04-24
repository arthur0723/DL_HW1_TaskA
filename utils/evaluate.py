import os
import torch
import numpy as np
import json
from torch.profiler import profile, record_function, ProfilerActivity
import time

def evaluate_model(model, dataloader, device='cuda', dataset_name='測試集'):
    """
    評估模型性能
    
    Args:
        model (nn.Module): 要評估的模型
        dataloader (DataLoader): 數據加載器
        device (str, optional): 設備 ('cuda' 或 'cpu')
        dataset_name (str, optional): 數據集名稱，用於顯示
    
    Returns:
        float: 模型準確率
    """
    print(f'在{dataset_name}上評估模型...')
    model.eval()
    running_corrects = 0
    
    # 遍歷數據
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # 前向傳播
        with torch.no_grad():
            try:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                # 統計正確預測數量
                running_corrects += torch.sum(preds == labels.data)
            except Exception as e:
                print(f"評估時發生錯誤: {e}")
                return 0.0
    
    # 計算準確率
    acc = running_corrects.double() / len(dataloader.dataset)
    print(f'{dataset_name}準確率: {acc:.4f}')
    
    return acc.item()

def calculate_params(model):
    """
    計算模型的參數數量
    
    Args:
        model (nn.Module): 要分析的模型
    
    Returns:
        int: 參數數量
    """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型參數數量: {params:,}")
    
    return params

def calculate_flops(model, dataloader, num_batches=1):
    """
    估算模型的浮點運算數量 (FLOPs)
    
    Args:
        model (nn.Module): 要分析的模型
        dataloader (DataLoader): 數據加載器
        num_batches (int, optional): 要分析的批次數量
    
    Returns:
        int: 估算的 FLOPs
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 使用第一個批次進行分析
    inputs, _ = next(iter(dataloader))
    inputs = inputs.to(device)
    
    try:
        from thop import profile, clever_format
        # 使用 thop 計算 FLOPs 和參數量
        flops, params = profile(model, inputs=(inputs,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        print(f"估算的 FLOPs: {flops}")
        return flops
    except ImportError:
        # 如果沒有安裝 thop，使用簡單的估算方法
        # 基於模型參數量和輸入大小估算
        total_params = sum(p.numel() for p in model.parameters())
        batch_size, channels, height, width = inputs.shape
        
        # 假設每個參數在前向傳播中使用一次
        # 這是一個非常粗略的估算
        flops = total_params * height * width
        print(f"估算的 FLOPs (粗略): {flops:,}")
        return flops

def save_evaluation_results(results, save_dir):
    """
    保存評估結果
    
    Args:
        results (dict): 評估結果字典
        save_dir (str): 保存目錄
    """
    # 保存為JSON文件
    with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"評估結果已保存至 {os.path.join(save_dir, 'evaluation_results.json')}")
    
    # 創建一個摘要文件，便於查看
    with open(os.path.join(save_dir, 'results_summary.txt'), 'w') as f:
        for key, value in results.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")