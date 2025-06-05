import os
import time
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR

from data.data_processing import batch_cwt
from data.data_loader import get_source_data
from data.data_augmentation import interaug
from utilities.plotting import plot_metrics, log_metrics
from utilities.utils import EarlyStopping 
from models.feature_fusion import Decision_Fusion
from test import evaluate

gpus = [0]


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion_cls, 
                    original_data, original_label, batch_size, signal_length, 
                    device, behavioral_features, freq_min, freq_max, tensor_height, 
                    sampling_frequency):
    start_time = time.time()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    total_samples = 0

    model.train()

    for _, (train_signal_data, train_label) in enumerate(tqdm(dataloader, desc="Processing")):
        # Data augmentation on the fly and concatenation with the existing batch
        aug_signal_data, aug_label = interaug(
            original_data, original_label, batch_size, 
            signal_length, device, num_behavioral_features=len(behavioral_features))
        train_signal_data = torch.cat((train_signal_data, aug_signal_data))
        train_label = torch.cat((train_label, aug_label))

        # Apply Continuous Wavelet Transform (CWT) to the data
        frequencies = np.linspace(freq_min, freq_max, tensor_height)
        train_cwt_data = batch_cwt(train_signal_data, frequencies, sampling_frequency=sampling_frequency)

        # Forward pass through the model with both original and CWT data
        outputs = model(train_signal_data, train_cwt_data)
        loss = criterion_cls(outputs, train_label)
        total_loss += loss.item()
        num_batches += 1

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == train_label).sum().item()
        total_samples += train_label.size(0)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    epoch_loss = total_loss / num_batches
    train_acc = total_correct / total_samples
    end_time = time.time()
    duration = end_time - start_time

    return epoch_loss, train_acc, duration, current_lr


def infer_ts_in_dim(train_folder_path):
    """自動推斷 TS head 輸入維度（根據 segment 長度）"""
    for fname in os.listdir(train_folder_path):
        if fname.endswith('.csv'):
            df = pd.read_csv(os.path.join(train_folder_path, fname))
            seg_len = df.shape[0]
            # 以你的 pipeline，280 對應 520, 560 對應 1240
            if seg_len == 280:
                return 520
            elif seg_len == 560:
                return 1240
            else:
                # 若是直接展平成 520/1240 維，可直接返回 seg_len
                return seg_len
    raise ValueError(f"無法推斷 segment 輸入維度於 {train_folder_path}")


def train(n_classes, batch_size, b1, b2, n_epochs, lr, behavioral_features,
          train_folder_path, test_folder_path, label_file, milestones, gamma,
          patience, sampling_frequency, weight_decay, freq_min, freq_max, tensor_height):
    """
    Train the model and evaluate it on the test set.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion_cls = torch.nn.CrossEntropyLoss().to(device)

    # >>>>> 根據資料自動推斷 TS head 輸入維度 <<<<<
    ts_in_dim = infer_ts_in_dim(train_folder_path)
    print(f"[模型設定] TS head 輸入維度自動推斷為 {ts_in_dim}")

    model = Decision_Fusion(n_classes, ts_in_dim=ts_in_dim)
    model = nn.DataParallel(model)
    model = model.to(device)

    # Find the length of the behavioral signal
    first_file_path = os.path.join(train_folder_path, os.listdir(train_folder_path)[0])
    signal_length = len(pd.read_csv(first_file_path))

    # Load the data from training and test directories and preserve the original for augmentation
    train_signal_data, train_label, test_signal_data, test_label = get_source_data(
        train_folder_path, test_folder_path, label_file, behavioral_features)
    original_data = train_signal_data
    original_label = train_label

    # Convert the data into PyTorch tensors and transfer to the appropriate device
    train_signal_data = torch.from_numpy(train_signal_data).float().to(device)
    train_label = torch.from_numpy(train_label).long().to(device)
    test_signal_data = torch.from_numpy(test_signal_data).float().to(device)
    test_label = torch.from_numpy(test_label).long().to(device)

    # Create a data loader for training data
    dataset = torch.utils.data.TensorDataset(train_signal_data, train_label)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # Setup the optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # Initialize early stopping mechanism
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    print(f'\nTrain set size: {train_label.shape[0]}')
    print(f'Test set size: {test_label.shape[0]}')
    print(f'\nTraining TCCT-Net...')
    print(f'Training on device: {device}')

    # Initialize lists to keep track of training and test performance
    best_acc = 0
    train_losses, train_accuracies, test_accuracies = [], [], []

    for e in range(n_epochs):
        print('\nEpoch:', e + 1)

        # Training for one epoch
        epoch_loss, train_acc, duration, current_lr = train_one_epoch(
            model, dataloader, optimizer, scheduler, criterion_cls, original_data, original_label,
            batch_size, signal_length, device, behavioral_features, freq_min, freq_max,
            tensor_height, sampling_frequency)

        # Evaluate on test set and log metrics
        test_acc, loss_test, y_pred = evaluate(
            model, test_signal_data, test_label, criterion_cls, freq_min, freq_max, tensor_height, sampling_frequency)
        log_metrics(e, epoch_loss, loss_test, train_acc, test_acc, best_acc, duration, current_lr)

        if test_acc > best_acc:
            best_acc = test_acc

        train_losses.append(epoch_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        torch.cuda.empty_cache()

        early_stopping(test_acc)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    torch.save(model.module.state_dict(), 'model_weights.pth')
    print('The best accuracy is:', best_acc)
    plot_metrics(train_losses, train_accuracies, test_accuracies)

    return test_label, y_pred, train_losses, None, train_accuracies, test_accuracies

# import os
# import time
# import torch
# import numpy as np
# from tqdm import tqdm
# import pandas as pd
# from torch import nn
# from torch.optim.lr_scheduler import MultiStepLR

# from data.data_processing import batch_cwt
# from data.data_loader import get_source_data
# from data.data_augmentation import interaug
# from utilities.plotting import plot_metrics, log_metrics
# from utilities.utils import EarlyStopping
# from models.feature_fusion import Decision_Fusion
# from test import evaluate

# gpus = [0]

# def check_tensor_valid(tensor, name):
#     """檢查Tensor是否含nan/inf及基本數值範圍"""
#     if isinstance(tensor, torch.Tensor):
#         if torch.isnan(tensor).any():
#             print(f"[異常] {name} 含有 nan")
#             raise ValueError(f"{name} 含有 nan")
#         if torch.isinf(tensor).any():
#             print(f"[異常] {name} 含有 inf")
#             raise ValueError(f"{name} 含有 inf")
#         if tensor.is_floating_point():
#             print(f"[檢查] {name} min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}")
#         else:
#             print(f"[檢查] {name} min={tensor.min().item()}, max={tensor.max().item()}, dtype={tensor.dtype}")
#     elif isinstance(tensor, np.ndarray):
#         if np.isnan(tensor).any():
#             print(f"[異常] {name} 含有 nan (numpy)")
#             raise ValueError(f"{name} 含有 nan (numpy)")
#         if np.isinf(tensor).any():
#             print(f"[異常] {name} 含有 inf (numpy)")
#             raise ValueError(f"{name} 含有 inf (numpy)")
#         if np.issubdtype(tensor.dtype, np.floating):
#             print(f"[檢查] {name} min={np.min(tensor):.4f}, max={np.max(tensor):.4f}, mean={np.mean(tensor):.4f}, std={np.std(tensor):.4f}")
#         else:
#             print(f"[檢查] {name} min={np.min(tensor)}, max={np.max(tensor)}, dtype={tensor.dtype}")

# def train_one_epoch(model, dataloader, optimizer, scheduler, criterion_cls, 
#                     original_data, original_label, batch_size, signal_length, 
#                     device, behavioral_features, freq_min, freq_max, tensor_height, 
#                     sampling_frequency):
#     """
#     執行單一訓練 epoch，包含資料檢查、增強、CWT與前向/反向傳播。
#     """
#     start_time = time.time()
#     total_loss = 0
#     num_batches = 0
#     total_correct = 0
#     total_samples = 0
#     model.train()

#     for _, (train_signal_data, train_label) in enumerate(tqdm(dataloader, desc="Processing")):
#         # 原始 batch 檢查
#         check_tensor_valid(train_signal_data, "train_signal_data (原始batch)")
#         check_tensor_valid(train_label, "train_label (原始batch)")

#         # Augmentation
#         aug_signal_data, aug_label = interaug(
#             original_data, original_label, batch_size, 
#             signal_length, device, num_behavioral_features=len(behavioral_features))
#         check_tensor_valid(aug_signal_data, "aug_signal_data")
#         check_tensor_valid(aug_label, "aug_label")

#         # 合併
#         train_signal_data = torch.cat((train_signal_data, aug_signal_data))
#         train_label = torch.cat((train_label, aug_label))
#         check_tensor_valid(train_signal_data, "train_signal_data (合併後)")
#         check_tensor_valid(train_label, "train_label (合併後)")

#         # CWT
#         frequencies = np.linspace(freq_min, freq_max, tensor_height)
#         train_cwt_data = batch_cwt(train_signal_data, frequencies, sampling_frequency=sampling_frequency)
#         check_tensor_valid(train_cwt_data, "train_cwt_data")

#         # 前向傳播
#         outputs = model(train_signal_data, train_cwt_data)
#         check_tensor_valid(outputs, "model outputs")

#         # Loss
#         assert train_label.dtype == torch.long, f"train_label dtype 應為 long, 實際為 {train_label.dtype}"
#         assert outputs.shape[0] == train_label.shape[0], "outputs 與 train_label batch size 不符"
#         loss = criterion_cls(outputs, train_label)
#         if torch.isnan(loss) or torch.isinf(loss):
#             print("[異常] 計算loss時發生 nan/inf")
#             print(f"outputs: {outputs}")
#             print(f"train_label: {train_label}")
#             raise ValueError("loss 出現 nan 或 inf")
#         total_loss += loss.item()
#         num_batches += 1

#         # 準確率計算
#         _, predicted = torch.max(outputs.data, 1)
#         total_correct += (predicted == train_label).sum().item()
#         total_samples += train_label.size(0)

#         # 反向傳播與優化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     scheduler.step()
#     current_lr = scheduler.get_last_lr()[0]
#     epoch_loss = total_loss / num_batches
#     train_acc = total_correct / total_samples

#     end_time = time.time()
#     duration = end_time - start_time

#     return epoch_loss, train_acc, duration, current_lr

# def train(n_classes, batch_size, b1, b2, n_epochs, lr, behavioral_features,
#           train_folder_path, test_folder_path, label_file, milestones, gamma,
#           patience, sampling_frequency, weight_decay, freq_min, freq_max, tensor_height):
#     """
#     主訓練流程（包含資料讀取與預檢查）。
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 載入資料（含 nan 清除，檔名輸出皆於 get_source_data 內實作）
#     train_signal_data, train_label, test_signal_data, test_label = get_source_data(
#         train_folder_path, test_folder_path, label_file, behavioral_features)
#     check_tensor_valid(train_signal_data, "train_signal_data (載入)")
#     check_tensor_valid(train_label, "train_label (載入)")
#     check_tensor_valid(test_signal_data, "test_signal_data (載入)")
#     check_tensor_valid(test_label, "test_label (載入)")

#     original_data = train_signal_data
#     original_label = train_label

#     train_signal_data = torch.from_numpy(train_signal_data).float().to(device)
#     train_label = torch.from_numpy(train_label).long().to(device)
#     test_signal_data = torch.from_numpy(test_signal_data).float().to(device)
#     test_label = torch.from_numpy(test_label).long().to(device)

#     dataset = torch.utils.data.TensorDataset(train_signal_data, train_label)
#     dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

#     model = Decision_Fusion(n_classes)
#     model = nn.DataParallel(model)
#     model = model.to(device)
#     criterion_cls = torch.nn.CrossEntropyLoss().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
#     scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
#     early_stopping = EarlyStopping(patience=patience, verbose=True)

#     best_acc = 0
#     train_losses, train_accuracies, test_accuracies = [], [], []

#     print(f'\nTrain set size: {train_label.shape[0]}')
#     print(f'Test set size: {test_label.shape[0]}')
#     print(f'\nTraining TCCT-Net...')
#     print(f'Training on device: {device}')

#     for e in range(n_epochs):
#         print('\nEpoch:', e + 1)
#         epoch_loss, train_acc, duration, current_lr = train_one_epoch(
#             model, dataloader, optimizer, scheduler, criterion_cls, original_data, original_label,
#             batch_size, train_signal_data.shape[-1], device, behavioral_features, freq_min, freq_max,
#             tensor_height, sampling_frequency)
#         test_acc, loss_test, y_pred = evaluate(
#             model, test_signal_data, test_label, criterion_cls, freq_min, freq_max, tensor_height, sampling_frequency)
#         log_metrics(e, epoch_loss, loss_test, train_acc, test_acc, best_acc, duration, current_lr)
#         if test_acc > best_acc:
#             best_acc = test_acc
#         train_losses.append(epoch_loss)
#         train_accuracies.append(train_acc)
#         test_accuracies.append(test_acc)
#         torch.cuda.empty_cache()
#         early_stopping(test_acc)
#         if early_stopping.early_stop:
#             print("Early stopping")
#             break

#     torch.save(model.module.state_dict(), 'model_weights.pth')
#     print('The best accuracy is:', best_acc)
#     plot_metrics(train_losses, train_accuracies, test_accuracies)
#     return test_label, y_pred
