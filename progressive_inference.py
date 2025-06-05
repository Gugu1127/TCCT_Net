import json
import torch
import random
import numpy as np
import pandas as pd
from torch import nn
from collections import OrderedDict

from models.feature_fusion import Decision_Fusion
from test import evaluate
from data.data_loader import get_source_data_inference

import argparse

gpus = [0]

def inference(n_classes, behavioral_features, target_mean, target_std, inference_folder_path, label_file_inference,
              sampling_frequency, freq_min, freq_max, tensor_height, model_weights_path):
    """
    使用訓練時儲存的 mean/std 進行資料標準化並推論
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入模型權重
    model = Decision_Fusion(n_classes)
    ckpt = torch.load(model_weights_path)
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        new_key = 'module.' + k
        new_ckpt[new_key] = v

    model = nn.DataParallel(model).to(device)
    model.load_state_dict(new_ckpt)
    model.eval()

    # 載入與標準化推論資料
    inference_signal_data, inference_label = get_source_data_inference(
        inference_folder_path, label_file_inference, behavioral_features, target_mean, target_std)
    inference_signal_data = torch.from_numpy(inference_signal_data).float().to(device)
    inference_label = torch.from_numpy(inference_label).long().to(device)

    # 推論與評估
    criterion_cls = torch.nn.CrossEntropyLoss().to(device)
    inference_acc, loss_inference, y_pred = evaluate(
        model, inference_signal_data, inference_label, criterion_cls,
        freq_min, freq_max, tensor_height, sampling_frequency
    )

    print(f'Inference Accuracy: {inference_acc:.6f}')
    print(f'Inference Loss: {loss_inference:.6f}')

    return inference_label, y_pred

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# 設定隨機種子
seed_n = 42
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)
torch.cuda.manual_seed_all(seed_n)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference Script')
    parser.add_argument('--inference_folder_path', type=str, default='inference')
    parser.add_argument('--label_file_inference', type=str, default='label_inference.csv')
    parser.add_argument('--config', type=str, default='config.json', help='Config file path')
    args = parser.parse_args()

    # 載入最新 config，確保 mean/std 為訓練完自動更新值
    config = load_config(args.config)
    config['inference_folder_path'] = args.inference_folder_path
    config['label_file_inference'] = args.label_file_inference

    # 執行推論
    y_true, y_pred = inference(
        n_classes=config['n_classes'],
        behavioral_features=config['behavioral_features'],
        target_mean=config['target_mean'],
        target_std=config['target_std'],
        inference_folder_path=config['inference_folder_path'],
        label_file_inference=config['label_file_inference'],
        sampling_frequency=config['sampling_frequency'],
        freq_min=config['freq_min'],
        freq_max=config['freq_max'],
        tensor_height=config['tensor_height'],
        model_weights_path=config['final_model_weights']
    )

    # 儲存推論結果
    results_df = pd.DataFrame({
        'Actual Label': y_true.cpu().numpy(),
        'Predicted Label': y_pred.cpu().numpy()
    })
    results_df.to_csv('inference_results.csv', index=False)
    print('\nResults saved to inference_results.csv')
