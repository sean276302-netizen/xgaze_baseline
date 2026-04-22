import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.multiprocessing as mp
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

script_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(script_dir, 'src'))

from loss.def_loss import AngularLoss
from config_module import config_file
from dataloader.load_dataset import train_loader, val_loader, test_loader
from model.def_model import gaze_network_STN

def save_model(model, optimizer, epoch, save_dir, filename="model.pth"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 如果保存目录不存在，创建目录
    save_path = os.path.join(save_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    print(f"Model saved to {save_path}")

def load_model(model, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return model

def main():
    config = config_file.config_class()
    # 初始化模型和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.train.model == "gaze_network_STN":
        model = gaze_network_STN().to(device)
    criterion = AngularLoss()  # 使用角度损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    num_epochs = config.train.num_epochs

    '''if config.train.model_path:
        model, optimizer, epoch = load_model(model, optimizer, config.train.model_path)'''

    #以年月日时分命名模型保存目录
    save_dir = f"src/weights/{time.strftime('%Y_%m_%d_%H_%M')}"

    # 初始化学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config.train.lr_patience)

    scaler = GradScaler(enabled=True)

    # 初始化早停机制的参数
    early_stopping_patience = config.train.early_stopping_patience  # 容忍的epoch数量
    early_stopping_counter = 0  # 早停计数器
    best_val_loss = float('inf')  # 初始化最佳验证损失

    print(f"Training on {device}")
    print(f"Train samples: {len(train_loader.dataset)}\nVal samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        p1 = time.time()
        model.train()

        train_losses = []

        # 使用 tqdm 包装 train_loader，添加进度条
        progress_bar = tqdm(train_loader, desc=f"Training {epoch + 1}/{num_epochs}", unit="batch")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # 使用混合精度训练
            with autocast(device_type="cuda", enabled=True):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # 缩放梯度
            scaler.scale(loss).backward()

            # 更新权重与缩放器
            scaler.step(optimizer)
            scaler.update()

            '''outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()'''
            # 更新进度条的后缀信息，显示当前 batch 的 loss
            progress_bar.set_postfix({"Batch Loss": f"{loss.item():.2f}"})

            # 检查是否有 nan 损失
            if torch.isnan(loss):
                print("NaN loss detected")
                break
            #optimizer.step()

            # 记录每个 batch 的损失值
            train_losses.append(loss.item())

        train_losses = np.array(train_losses)
        # 计算训练阶段的平均损失和标准差
        train_mean_loss = np.mean(train_losses)
        train_std_loss = np.std(train_losses, ddof=0)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Mean Loss: {train_mean_loss:.2f}, Std: {train_std_loss:.2f}")

        # 验证过程
        model.eval()
        val_loss = 0
        val_losses = []
        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc=f"Validating {epoch + 1}/{num_epochs}", unit="batch")
            for i, (inputs, targets) in enumerate(progress_bar_val):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                val_losses.append(criterion(outputs, targets).item())
                progress_bar_val.set_postfix({"Mean Loss": f"{val_loss/(i+1):.2f}"})
        val_loss /= len(val_loader)
        # 计算验证阶段的平均损失和标准差
        val_losses = np.array(val_losses)
        val_std_loss = np.std(val_losses, ddof=0)
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Mean Loss: {val_loss:.2f}, Std: {val_std_loss:.2f}")

        p2 = time.time()
        print(f"Epoch [{epoch+1}/{num_epochs}], time: {(p2-p1)//60:.0f}m {(p2-p1)%60:.0f}s\n")

        # 更新学习率
        scheduler.step(val_loss)

        # 检查是否需要早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0  # 重置计数器
            save_model(model, optimizer, epoch, save_dir, 
                       filename=f"best_model_{best_val_loss:.2f}.pth")
            print(f"Best model saved, Validation Mean Loss: {best_val_loss:.2f}, Std Loss: {val_std_loss:.2f}\n")
        else:
            print(f"Validation Loss did not improve from {best_val_loss:.2f}, early stopping counter: [{early_stopping_counter + 1}/{config.train.early_stopping_patience}]\n")
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs\n")
                break

    model = load_model(model, os.path.join(save_dir, f"best_model_{best_val_loss:.2f}.pth"))
    model.eval()
    test_losses = []
    with torch.no_grad():
        progress_bar_test = tqdm(test_loader, desc=f"Testing", unit="batch")
        for inputs, targets in progress_bar_test:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_losses.append(loss.item())
    
    test_losses = np.array(test_losses)
    # 计算测试阶段的平均损失和标准差
    test_mean_loss = np.mean(test_losses)
    test_std_loss = np.std(test_losses, ddof=0)
    print(f"Test Mean Loss: {test_mean_loss:.2f}, Std: {test_std_loss:.2f}")

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    main()