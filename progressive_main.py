"""
Main script to train the model and save the results.

This script sets the seed for reproducibility, trains the model using the `train` function from `train.py`,
and updates target_mean and target_std in the config file after training.
"""

import json
import torch
import random
import numpy as np
from progressive_train import train
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Progressive Train Main')
    parser.add_argument('--config', type=str, default='/home/gugu/TCCT_Net/config/progressive_config.json', help='Config file path')
    parser.add_argument('--train_folder_path', type=str, help='訓練資料夾 (覆蓋 config)')
    parser.add_argument('--label_file', type=str, help='label csv 路徑 (覆蓋 config)')
    parser.add_argument('--model_save_path', type=str, help='模型儲存路徑 (覆蓋 config)')
    parser.add_argument('--pretrained_weights', type=str, default=None, help='(選填)預訓練權重')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Setting the seed for reproducibility
seed_n = 42
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)
torch.cuda.manual_seed_all(seed_n)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    config = load_config(config_path)
    # 覆蓋路徑
    if args.train_folder_path:
        config['train_folder_path'] = args.train_folder_path
    if args.label_file:
        config['label_file'] = args.label_file
    if args.model_save_path:
        config['final_model_weights'] = args.model_save_path
    if args.pretrained_weights:
        pretrained_weights = args.pretrained_weights
    else:
        pretrained_weights = None


    # 執行訓練
    train_losses, target_mean, target_std = train(
        n_classes=config['n_classes'],
        batch_size=config['batch_size'],
        b1=config['b1'],
        b2=config['b2'],
        n_epochs=config['n_epochs'],
        lr=config['lr'],
        behavioral_features=config['behavioral_features'],
        train_folder_path=config['train_folder_path'],
        label_file=config['label_file'],
        milestones=config['milestones'],
        gamma=config['gamma'],
        sampling_frequency=config['sampling_frequency'],
        weight_decay=config['weight_decay'],
        freq_min=config['freq_min'],
        freq_max=config['freq_max'],
        tensor_height=config['tensor_height'],
        model_save_path=config['final_model_weights'],
        config_path=config_path,
        pretrained_weights=pretrained_weights  # <--- 這裡支援預訓練權重
    )

    print(f"\n訓練完成。最終 target_mean = {target_mean:.8f}, target_std = {target_std:.8f}")
    print("Config 已自動更新。最佳模型已儲存於:", config['final_model_weights'])