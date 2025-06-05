import os
import time
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR

from data.data_processing import batch_cwt
from data.data_loader import get_source_data
from data.data_augmentation import interaug
from models.feature_fusion import Decision_Fusion

gpus = [0]


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion_cls, 
                    original_data, original_label, batch_size, signal_length, 
                    device, behavioral_features, freq_min, freq_max, tensor_height, 
                    sampling_frequency):
    start_time = time.time()
    total_loss = 0
    num_batches = 0

    model.train()

    for _, (train_signal_data, train_label) in enumerate(tqdm(dataloader, desc="Processing")):
        # 嚴格取得 batch 的 signal_length
        current_signal_length = train_signal_data.shape[-1]
        aug_signal_data, aug_label = interaug(
            original_data, original_label, batch_size, 
            current_signal_length, device, num_behavioral_features=len(behavioral_features))
        train_signal_data = torch.cat((train_signal_data, aug_signal_data))
        train_label = torch.cat((train_label, aug_label))

        # CWT 轉換
        frequencies = np.linspace(freq_min, freq_max, tensor_height)
        train_cwt_data = batch_cwt(train_signal_data, frequencies, sampling_frequency=sampling_frequency)

        # 前向傳播與計算 loss
        outputs = model(train_signal_data, train_cwt_data)
        loss = criterion_cls(outputs, train_label)
        total_loss += loss.item()
        num_batches += 1

        # 反向傳播與優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    epoch_loss = total_loss / num_batches
    duration = time.time() - start_time

    return epoch_loss, duration, scheduler.get_last_lr()[0]


def infer_ts_in_dim(train_folder_path):
    for fname in os.listdir(train_folder_path):
        if fname.endswith('.csv'):
            df = pd.read_csv(os.path.join(train_folder_path, fname))
            seg_len = df.shape[0]
            if seg_len == 280:
                return 520
            elif seg_len == 560:
                return 1240
            else:
                return seg_len
    raise ValueError(f"無法推斷 segment 輸入維度於 {train_folder_path}")


def update_config_target_mean_std(config_path, target_mean, target_std):
    """自動更新 config.json 之 target_mean 與 target_std 欄位"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["target_mean"] = float(target_mean)
    config["target_std"] = float(target_std)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"[Config] 已自動更新 target_mean={target_mean:.8f}，target_std={target_std:.8f}")


def train(
    n_classes, batch_size, b1, b2, n_epochs, lr, behavioral_features,
    train_folder_path, label_file, milestones, gamma,
    sampling_frequency, weight_decay, freq_min, freq_max, tensor_height,
    model_save_path="best_model_weights.pth", config_path=None,
    pretrained_weights=None   # <--- 新增參數
):
    """
    全部資料訓練，不評估、不計算準確率，歷史最低 loss 自動存權重。
    訓練結束後，自動計算並更新 target_mean、target_std 至 config.json。
    可指定 pretrained_weights 以承接前次最佳模型。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion_cls = torch.nn.CrossEntropyLoss().to(device)

    ts_in_dim = infer_ts_in_dim(train_folder_path)
    print(f"[模型設定] TS head 輸入維度自動推斷為 {ts_in_dim}")

    model = Decision_Fusion(n_classes, ts_in_dim=ts_in_dim)
    model = nn.DataParallel(model).to(device)

    # ======== 支援導入前次最佳模型權重 ========
    if pretrained_weights is not None and os.path.isfile(pretrained_weights):
        try:
            state_dict = torch.load(pretrained_weights, map_location=device)
            # 若原模型無 "module." 前綴，補上
            if not any(k.startswith('module.') for k in state_dict.keys()):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dict['module.'+k] = v
                state_dict = new_state_dict
            model.load_state_dict(state_dict)
            print(f"[初始化] 已成功載入前次最佳模型權重: {pretrained_weights}")
        except Exception as e:
            print(f"[警告] 載入前次最佳模型失敗: {e}")
    else:
        print("[初始化] 未導入任何前次模型，進行全新訓練。")
    # =====================================

    # 載入訓練資料
    train_signal_data, train_label  = get_source_data(
        train_folder_path, None, label_file, behavioral_features)

    print(f"[資料載入] 訓練資料形狀: {train_signal_data.shape}, 標籤形狀: {train_label.shape}")
    print("所有資料 segment 長度分布：", set([d.shape[-1] for d in train_signal_data]))
    original_data = train_signal_data
    original_label = train_label

    # ========== 計算全資料之 mean/std ==========
    target_mean = float(np.mean(train_signal_data))
    target_std = float(np.std(train_signal_data))
    print(f"[統計] 計算得到 target_mean={target_mean:.8f}，target_std={target_std:.8f}")
    # ===========================================

    train_signal_data = torch.from_numpy(train_signal_data).float().to(device)
    train_label = torch.from_numpy(train_label).long().to(device)

    dataset = torch.utils.data.TensorDataset(train_signal_data, train_label)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    print(f"\n訓練樣本數: {train_label.shape[0]}")
    print(f"開始訓練，共 {n_epochs} epochs")

    best_loss = float('inf')
    best_epoch = -1
    train_losses = []

    for e in range(n_epochs):
        print(f"\nEpoch {e+1}/{n_epochs}")
        epoch_loss, duration, current_lr = train_one_epoch(
            model, dataloader, optimizer, scheduler, criterion_cls, original_data, original_label,
            batch_size, len(train_signal_data), device, behavioral_features, freq_min, freq_max,
            tensor_height, sampling_frequency)

        train_losses.append(epoch_loss)
        print(f"[Epoch {e+1}] Loss: {epoch_loss:.4f} | 時間: {duration:.2f}s | LR: {current_lr:.6f}")

        # 若為最佳 loss，儲存模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = e + 1
            if os.path.exists(model_save_path):
                print(f"[警告] model_save_path {model_save_path} 已存在，將被覆蓋。")
            torch.save(model.module.state_dict(), model_save_path)
            print(f"→ 新最佳模型已儲存於 '{model_save_path}'（epoch {best_epoch}, loss={best_loss:.6f}）")

        torch.cuda.empty_cache()

    print(f"\n訓練結束。最佳模型於第 {best_epoch} epoch, loss={best_loss:.6f}")

    # 訓練結束後自動更新 config.json 的 mean/std
    if config_path is not None:
        update_config_target_mean_std(config_path, target_mean, target_std)

    return train_losses, target_mean, target_std
