import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
import time

from models.standard_net import create_standard_net
from models.dynamic_net import create_dynamic_net
from utils.dataset import load_data
from utils.train import train_model, save_training_plots
from utils.evaluate import evaluate_model, calculate_params, calculate_flops, save_evaluation_results

def set_seed(seed=42):
    """
    設置隨機種子以獲得可重現的結果
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='Mini-ImageNet Classification - Task A')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                       help='Mode: train or evaluate')
    parser.add_argument('--model', type=str, default='dynamic', choices=['standard', 'dynamic'],
                       help='Model type: standard or dynamic')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint for evaluation')
    parser.add_argument('--channel_mode', type=str, default='rgb', 
                        choices=['rgb', 'r', 'g', 'b', 'rg', 'rb', 'gb'],
                        help='Channel mode for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data_dir', type=str, default='', help='Directory for image data')
    
    args = parser.parse_args()
    
    # 設置隨機種子
    set_seed(args.seed)
    
    # 設定裝置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    # 設定路徑
    data_dir = args.data_dir
    train_txt = 'train.txt'
    val_txt = 'val.txt'
    test_txt = 'test.txt'
    
    # 創建實驗目錄
    model_type = args.model
    model_save_dir = f'experiments/{model_type}'
    os.makedirs(model_save_dir, exist_ok=True)
    
    if args.mode == 'train':
        # 載入數據 - 訓練時使用標準RGB
        print("載入數據...")
        image_datasets, dataloaders, dataset_sizes, class_names = load_data(
            data_dir, train_txt, val_txt, test_txt, args.batch_size, channel_mode='rgb'
        )
        
        # 創建模型
        print(f"創建{model_type}模型...")
        if model_type == 'standard':
            model = create_standard_net(num_classes=class_names)
        else:  # dynamic
            model = create_dynamic_net(num_classes=class_names)
        
        model = model.to(device)
        
        # 定義損失函數和優化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
        )
        
        print(f"開始訓練模型，共 {args.epochs} 個周期...")
        
        # 訓練模型
        model, history = train_model(
            model, dataloaders, dataset_sizes, criterion, optimizer, scheduler,
            num_epochs=args.epochs, device=device
        )
        
        # 保存模型
        model_path = os.path.join(model_save_dir, 'best_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存至 {model_path}")
        
        # 保存訓練圖表
        save_training_plots(history, model_save_dir)
        
        # 計算模型參數數量和FLOPS
        params = calculate_params(model)
        flops = calculate_flops(model, dataloaders['test'])
        
        # 評估模型
        print("評估訓練後的模型...")
        test_acc = evaluate_model(model, dataloaders['test'], device, '測試集')
        
        # 整合結果
        results = {
            'model_type': model_type,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'test_accuracy': test_acc,
            'params': params,
            'flops': flops,
            'channel_mode': 'rgb'
        }
        
        # 保存評估結果
        save_evaluation_results(results, model_save_dir)
        
    else:  # Evaluation mode
        if not args.checkpoint:
            raise ValueError("Evaluation mode requires --checkpoint argument")
            
        print(f"載入檢查點 {args.checkpoint} 並評估模型，通道模式: {args.channel_mode}")
        
        # 載入數據 - 使用指定的通道模式
        print("載入數據...")
        image_datasets, dataloaders, dataset_sizes, class_names = load_data(
            data_dir, train_txt, val_txt, test_txt, args.batch_size, channel_mode=args.channel_mode
        )
        
        # 創建模型
        if model_type == 'standard':
            model = create_standard_net(num_classes=class_names)
        else:  # dynamic
            model = create_dynamic_net(num_classes=class_names)
            
        # 載入模型
        model.load_state_dict(torch.load(args.checkpoint))
        model = model.to(device)
        model.eval()
        
        # 評估模型
        train_acc = evaluate_model(model, dataloaders['train'], device, '訓練集')
        val_acc = evaluate_model(model, dataloaders['val'], device, '驗證集')
        test_acc = evaluate_model(model, dataloaders['test'], device, '測試集')
        
        # 計算模型參數數量和FLOPS
        params = calculate_params(model)
        flops = calculate_flops(model, dataloaders['test'])
        
        # 測量推論時間
        start_time = time.time()
        for _ in range(5):  # 多次測量取平均值
            for inputs, _ in dataloaders['test']:
                inputs = inputs.to(device)
                with torch.no_grad():
                    _ = model(inputs)
        
        inference_time = (time.time() - start_time) / (5 * len(dataloaders['test']))
        
        # 整合結果
        results = {
            'model_type': model_type,
            'channel_mode': args.channel_mode,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'params': params,
            'flops': flops,
            'inference_time': inference_time
        }
        
        # 保存評估結果
        eval_dir = os.path.join(model_save_dir, f'eval_{args.channel_mode}')
        os.makedirs(eval_dir, exist_ok=True)
        save_evaluation_results(results, eval_dir)
        
        print("評估完成!")
        print(f"測試集準確率: {test_acc:.4f}")
        print(f"模型參數數量: {params:,}")
        print(f"推論時間(每批次): {inference_time:.4f} 秒")

if __name__ == "__main__":
    main()