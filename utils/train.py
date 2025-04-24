import time
import copy
import os
import torch
import matplotlib.pyplot as plt

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler=None, 
                num_epochs=10, device='cuda'):
    """
    訓練模型
    
    Args:
        model (nn.Module): 要訓練的模型
        dataloaders (dict): 包含訓練和驗證數據加載器的字典
        dataset_sizes (dict): 包含訓練和驗證數據集大小的字典
        criterion (nn.Module): 損失函數
        optimizer (torch.optim.Optimizer): 優化器
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 學習率調度器
        num_epochs (int, optional): 訓練的周期數
        device (str, optional): 訓練設備 ('cuda' 或 'cpu')
    
    Returns:
        tuple: (best_model, history) - 最佳模型和訓練歷史
    """
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # 記錄訓練過程中的損失和準確率
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 每個 epoch 都有訓練和驗證階段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 設置模型為訓練模式
            else:
                model.eval()   # 設置模型為評估模式
                
            running_loss = 0.0
            running_corrects = 0
            
            # 遍歷數據
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 將參數梯度歸零
                optimizer.zero_grad()
                
                # 前向傳播
                # 只有在訓練時才跟踪歷史記錄
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # 如果是訓練階段，則反向傳播 + 優化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # 統計
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 記錄歷史
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # 對於 ReduceLROnPlateau 調度器，需要在驗證階段進行步進
                if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_loss)
            
            # 如果是驗證階段且性能更好，則保存模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        # 對於其他調度器，在每個 epoch 結束後進行步進
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
                
        print()
        
    time_elapsed = time.time() - since
    print(f'訓練完成，總耗時 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳驗證準確率: {best_acc:.4f}')
    
    # 載入最佳模型權重
    model.load_state_dict(best_model_wts)
    return model, history

def save_training_plots(history, save_dir):
    """
    保存訓練過程的損失和準確率曲線
    
    Args:
        history (dict): 包含訓練和驗證損失及準確率的字典
        save_dir (str): 保存圖表的目錄
    """
    plt.figure(figsize=(12, 5))
    
    # 繪製損失曲線
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 繪製準確率曲線
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()
    
    print(f"訓練歷史圖表已保存至 {os.path.join(save_dir, 'training_history.png')}")
    
    # 保存原始數據為 CSV
    import csv
    with open(os.path.join(save_dir, 'training_history.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
        for i in range(len(history['train_loss'])):
            writer.writerow([
                i+1, 
                history['train_loss'][i], 
                history['val_loss'][i], 
                history['train_acc'][i], 
                history['val_acc'][i]
            ])
    
    print(f"訓練歷史數據已保存至 {os.path.join(save_dir, 'training_history.csv')}")

def train_with_multiple_channel_modes(model, data_dir, train_txt, val_txt, test_txt, 
                                      criterion, optimizer, scheduler=None, 
                                      batch_size=64, num_epochs=10, device='cuda'):
    """
    使用多種通道模式輪流訓練模型（用於動態卷積模型）
    
    Args:
        model (nn.Module): 要訓練的模型
        data_dir (str): 數據目錄
        train_txt (str): 訓練集文本文件
        val_txt (str): 驗證集文本文件
        test_txt (str): 測試集文本文件
        criterion (nn.Module): 損失函數
        optimizer (torch.optim.Optimizer): 優化器
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 學習率調度器
        batch_size (int, optional): 批次大小
        num_epochs (int, optional): 訓練的周期數
        device (str, optional): 訓練設備 ('cuda' 或 'cpu')
    
    Returns:
        tuple: (trained_model, history) - 訓練後的模型和訓練歷史
    """
    from utils.dataset import load_data
    
    # 通道模式列表
    channel_modes = ['rgb', 'r', 'g', 'b', 'rg', 'rb', 'gb']
    
    # 總訓練歷史
    total_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 使用每種通道模式訓練一定數量的 epoch
    epochs_per_mode = max(1, num_epochs // len(channel_modes))
    remaining_epochs = num_epochs - (epochs_per_mode * len(channel_modes))
    
    for i, mode in enumerate(channel_modes):
        print(f"\n===== 使用通道模式 '{mode}' 訓練 {epochs_per_mode} 個周期 =====")
        
        # 為最後一個模式添加剩餘的周期
        current_epochs = epochs_per_mode
        if i == len(channel_modes) - 1:
            current_epochs += remaining_epochs
        
        # 載入該通道模式的數據
        image_datasets, dataloaders, dataset_sizes, _ = load_data(
            data_dir, train_txt, val_txt, test_txt, batch_size, channel_mode=mode
        )
        
        # 訓練模型
        model, history = train_model(
            model, dataloaders, dataset_sizes, criterion, optimizer, scheduler,
            num_epochs=current_epochs, device=device
        )
        
        # 添加到總歷史
        total_history['train_loss'].extend(history['train_loss'])
        total_history['train_acc'].extend(history['train_acc'])
        total_history['val_loss'].extend(history['val_loss'])
        total_history['val_acc'].extend(history['val_acc'])
    
    return model, total_history