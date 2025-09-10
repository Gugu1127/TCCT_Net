import json
import torch
import random
import numpy as np
import pandas as pd
from torch import nn
from collections import OrderedDict

from models.feature_fusion import Decision_Fusion
from data.data_loader import get_source_data_inference

import argparse

def inference(n_classes, behavioral_features, target_mean, target_std,
              inference_folder_path, sampling_frequency, freq_min, freq_max,
              tensor_height, model_weights_path):
    """
    僅做推理（不需要 label），儲存預測結果
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Decision_Fusion(n_classes)
    ckpt = torch.load(model_weights_path)
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        new_key = 'module.' + k
        new_ckpt[new_key] = v

    model = nn.DataParallel(model).to(device)
    model.load_state_dict(new_ckpt)
    model.eval()

    # 只載入資料（無需 label）
    inference_signal_data = get_source_data_inference(
        inference_folder_path, behavioral_features, target_mean, target_std
    )
    # 推理時不用 label
    inference_signal_data = torch.from_numpy(inference_signal_data).float().to(device)

    # 推論
    with torch.no_grad():
        # 可依您的模型輸入調整
        frequencies = np.linspace(freq_min, freq_max, tensor_height)
        from data.data_processing import batch_cwt
        cwt_data = batch_cwt(inference_signal_data, frequencies, sampling_frequency=sampling_frequency)
        outputs = model(inference_signal_data, cwt_data)
        pred_label = torch.argmax(outputs, dim=1)
    return pred_label

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

# -- main --
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Progressive Inference')
    parser.add_argument('--inference_folder_path', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model_weights_path', type=str, default=None, help='預訓練模型路徑（優先於 config）')
    parser.add_argument('--output_csv', type=str, default='inference_pred.csv', help='預測結果輸出檔名')
    args = parser.parse_args()

    # 設定 seed
    seed_n = 42
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = load_config(args.config)
    # 決定要用哪個 model weights
    model_weights_path = args.model_weights_path or config.get('final_model_weights')
    assert model_weights_path is not None, "請提供模型權重檔路徑！"

    pred_label = inference(
        n_classes=config['n_classes'],
        behavioral_features=config['behavioral_features'],
        target_mean=config['target_mean'],
        target_std=config['target_std'],
        inference_folder_path=args.inference_folder_path,
        sampling_frequency=config['sampling_frequency'],
        freq_min=config['freq_min'],
        freq_max=config['freq_max'],
        tensor_height=config['tensor_height'],
        model_weights_path=model_weights_path
    )

    # 輸出預測
    df = pd.DataFrame({'Predicted Label': pred_label.cpu().numpy()})
    df.to_csv(args.output_csv, index=False)
    print(f"預測結果已輸出至 {args.output_csv}")
