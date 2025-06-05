import os
import json
import shutil
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from train import train  # 需確保 train.py 於同目錄或 PYTHONPATH

VERSION_CONFIGS = [
    {"size": 280, "ver": "V1",    "no0": False},
    {"size": 560, "ver": "V1",    "no0": False},
    {"size": 280, "ver": "V2",    "no0": False},
    {"size": 560, "ver": "V2",    "no0": False},
    {"size": 280, "ver": "V1",    "no0": True},
    {"size": 560, "ver": "V1",    "no0": True},
    {"size": 280, "ver": "V2",    "no0": True},
    {"size": 560, "ver": "V2",    "no0": True},
]

BASE_DATASET_DIR = "/home/gugu/TCCT_Net/output_dataset"
BASE_RESULT_DIR  = "/home/gugu/TCCT_Net/train_results"
BASE_LABELS_DIR  = "/home/gugu/TCCT_Net/output_dataset/labels"
ORIGINAL_CONFIG_PATH = "/home/gugu/TCCT_Net/config/config.json"

def set_seed(seed_n=42):
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)

def version_name(cfg):
    base = f"dataset_{cfg['size']}_{cfg['ver']}"
    if cfg['no0']:
        base += "_no0"
    return base

def update_config_for_version(base_cfg, v_cfg, data_root, labels_root):
    ver_suffix = f"{v_cfg['size']}_{v_cfg['ver']}{'_no0' if v_cfg['no0'] else ''}"
    train_path = os.path.join(data_root, f"train_{ver_suffix}")
    test_path  = os.path.join(data_root, f"test_{ver_suffix}")
    label_file = os.path.join(labels_root, f"{ver_suffix}_label.csv")
    new_cfg = dict(base_cfg)
    new_cfg["train_folder_path"] = train_path
    new_cfg["test_folder_path"]  = test_path
    new_cfg["label_file"]        = label_file
    new_cfg["n_classes"]         = 4 if v_cfg['no0'] else 4
    return new_cfg

def save_training_log(log_path, train_loss, train_acc, test_acc):
    log_df = pd.DataFrame({
        "epoch": list(range(1, len(train_loss) + 1)),
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_acc": test_acc
    })
    log_df.to_csv(log_path, index=False)
    print(f"已儲存訓練紀錄於 {log_path}")

def setup_logger(log_path):
    # 建立 logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 清理舊 handler，避免多重輸出
    if logger.hasHandlers():
        logger.handlers.clear()
    # 設定 FileHandler
    handler = logging.FileHandler(log_path, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # 同時也輸出到螢幕（可選）
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger

def main():
    set_seed(42)
    os.makedirs(BASE_RESULT_DIR, exist_ok=True)
    base_cfg = load_config(ORIGINAL_CONFIG_PATH)
    for v_idx, v_cfg in enumerate(VERSION_CONFIGS):
        # 建立版本輸出目錄
        result_dir = os.path.join(BASE_RESULT_DIR, version_name(v_cfg))
        os.makedirs(result_dir, exist_ok=True)
        # 每版本指定 log 檔
        log_path = os.path.join(result_dir, "train.log")
        logger = setup_logger(log_path)
        logger.info("="*30)
        logger.info(f"[{v_idx+1}/8] 訓練版本：{version_name(v_cfg)}")
        # 動態產生當前版本 config
        cfg = update_config_for_version(base_cfg, v_cfg, BASE_DATASET_DIR, BASE_LABELS_DIR)
        save_config(cfg, os.path.join(result_dir, "config_used.json"))
        try:
            # train() 需回傳 y_true, y_pred, train_loss, None, train_acc, test_acc
            y_true, y_pred, train_loss, _, train_acc, test_acc = train(
                n_classes=cfg['n_classes'],
                batch_size=cfg['batch_size'],
                b1=cfg['b1'],
                b2=cfg['b2'],
                n_epochs=cfg['n_epochs'],
                lr=cfg['lr'],
                behavioral_features=cfg['behavioral_features'],
                train_folder_path=cfg['train_folder_path'],
                test_folder_path=cfg['test_folder_path'],
                label_file=cfg['label_file'],
                milestones=cfg['milestones'],
                gamma=cfg['gamma'],
                patience=cfg['patience'],
                sampling_frequency=cfg['sampling_frequency'],
                weight_decay=cfg['weight_decay'],
                freq_min=cfg['freq_min'],
                freq_max=cfg['freq_max'],
                tensor_height=cfg['tensor_height']
            )
            # 儲存測試結果
            result_csv_path = os.path.join(result_dir, 'test_results.csv')
            pd.DataFrame({
                'Actual Label': y_true.cpu().numpy(),
                'Predicted Label': y_pred.cpu().numpy()
            }).to_csv(result_csv_path, index=False)
            logger.info(f"已儲存測試結果於 {result_csv_path}")

            # 儲存訓練歷程紀錄
            train_log_csv_path = os.path.join(result_dir, 'training_log.csv')
            save_training_log(train_log_csv_path, train_loss, train_acc, test_acc)
            logger.info(f"已儲存訓練紀錄於 {train_log_csv_path}")

        except Exception as e:
            logger.error(f"訓練過程出錯：{e}", exc_info=True)
    print("\n全部 8 種版本訓練與結果儲存已完成！")
    
    
if __name__ == "__main__":
    main()
