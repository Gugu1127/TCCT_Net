import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_csv_data(folder_path, label_file, behavioral_features):
    """
    Load data from CSV files in a folder and corresponding labels.

    Args:
        folder_path (str): Path to the folder containing the CSV files.
        label_file (str): Path to the CSV file containing the labels.
        behavioral_features (list of str): List of behavioral features to extract from the data.

    Returns:
        np.array: Array of data extracted from the CSV files.
        np.array: Array of labels corresponding to the data.
    """
        
    if os.path.getsize(label_file) == 0:
        raise ValueError(f"標籤檔 {label_file} 為空，請檢查資料前處理流程！")
    labels_df = pd.read_csv(label_file)
    all_data, all_labels = [], []

    for filename in tqdm(os.listdir(folder_path), desc="Loading data"):
        if filename.endswith('.csv'):
            subject_id = filename.split('.')[0]
            subject_file = os.path.join(folder_path, filename)
            if os.path.getsize(subject_file) == 0:
                print(f"[警告] 檔案為空檔，已跳過: {subject_file}")
                continue
            try:
                subject_data = pd.read_csv(subject_file)
            except pd.errors.EmptyDataError:
                print(f"[警告] 檔案無法解析為有效 CSV，已跳過: {subject_file}")
                continue

            # 欄位一致性檢查
            missing_cols = [col for col in behavioral_features if col not in subject_data.columns]
            if missing_cols:
                print(f"[警告] 檔案 {subject_file} 缺少特徵欄位: {missing_cols}，已跳過。")
                continue

            # 空段資料檢查
            if any(len(subject_data[col].values) == 0 for col in behavioral_features):
                print(f"[警告] 檔案 {subject_file} 中部分特徵為空，已跳過。")
                continue

            subject_data_values = np.stack([subject_data[col].values for col in behavioral_features], axis=0)
            subject_label = labels_df[labels_df['chunk'].str.contains(subject_id)]['label'].values

            if len(subject_label) > 0:
                all_data.append(subject_data_values)
                all_labels.append(subject_label[0])
            else:
                print(f"No label found for subject {subject_id}")

    if len(all_data) == 0:
        raise ValueError(f"[嚴重警告] 未成功讀取任何有效資料，請檢查資料來源與標籤對應！")

    all_data = np.array(all_data)
    all_data = np.expand_dims(all_data, axis=1)  
    all_labels = np.array(all_labels)

    return all_data, all_labels


def get_source_data(train_folder_path, test_folder_path, label_file, behavioral_features):
    """
    Load and preprocess training and testing data from the specified folders.

    Args:
        train_folder_path (str): Path to the folder containing the training data.
        test_folder_path (str): Path to the folder containing the testing data.
        label_file (str): Path to the CSV file containing the labels.
        behavioral_features (list of str): List of behavioral features to extract from the data.

    Returns:
        np.array: Processed training data.
        np.array: Labels for the training data.
        np.array: Processed testing data.
        np.array: Labels for the testing data.
    """

    # Load training data
    print('\nLoading train data ...')
    train_data, train_labels = load_csv_data(train_folder_path, label_file, behavioral_features)
    train_labels = train_labels.reshape(1, -1)

    # Shuffle the training data
    shuffle_index = np.random.permutation(len(train_data))
    train_data = train_data[shuffle_index, :, :, :]
    train_labels = train_labels[0][shuffle_index]

    # Load testing data
    if test_folder_path is not None:
        print('\nLoading test data ...')
        test_data, test_labels = load_csv_data(test_folder_path, label_file, behavioral_features)
        test_labels = test_labels.reshape(-1)  
        # Standardize both train and test data using training data statistics
        target_mean = np.mean(train_data)
        target_std = np.std(train_data)
        train_data = (train_data - target_mean) / target_std
        test_data = (test_data - target_mean) / target_std
        return train_data, train_labels, test_data, test_labels
    else:
        test_data, test_labels = None, None
        # 只標準化 train
        target_mean = np.mean(train_data)
        target_std = np.std(train_data)
        train_data = (train_data - target_mean) / target_std
        return train_data, train_labels

    


def get_source_data_inference(inference_folder_path, label_file_inference,
                              behavioral_features, target_mean, target_std):
    """
    Load and preprocess inference data from the specified folder.

    Args:
        inference_folder_path (str): Path to the folder containing the inference data.
        label_file_inference (str): Path to the CSV file containing the labels.
        behavioral_features (list of str): List of behavioral features to extract from the data.
        target_mean (float): Mean value of the training data used for standardization.
        target_std (float): Standard deviation of the training data used for standardization.

    Returns:
        np.array: Processed testing data.
        np.array: Labels for the testing data.
    """

    # Load inference data
    print('\nLoading data for inference ...')
    inference_data, inference_labels = load_csv_data(inference_folder_path, label_file_inference, behavioral_features)
    inference_labels = inference_labels.reshape(-1)

    # Standardize inference data using provided training data statistics
    inference_data = (inference_data - target_mean) / target_std

    return inference_data, inference_labels


# import os
# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# def load_csv_data(folder_path, label_file, behavioral_features):
#     """
#     載入資料，回傳(data, labels, filenames)
#     """
#     if os.path.getsize(label_file) == 0:
#         raise ValueError(f"標籤檔 {label_file} 為空，請檢查資料前處理流程！")
#     labels_df = pd.read_csv(label_file)
#     all_data, all_labels, all_filenames = [], [], []

#     for filename in tqdm(os.listdir(folder_path), desc="Loading data"):
#         if filename.endswith('.csv'):
#             subject_id = filename.split('.')[0]
#             subject_file = os.path.join(folder_path, filename)
#             if os.path.getsize(subject_file) == 0:
#                 print(f"[警告] 檔案為空檔，已跳過: {subject_file}")
#                 continue
#             try:
#                 subject_data = pd.read_csv(subject_file)
#             except pd.errors.EmptyDataError:
#                 print(f"[警告] 檔案無法解析為有效 CSV，已跳過: {subject_file}")
#                 continue

#             missing_cols = [col for col in behavioral_features if col not in subject_data.columns]
#             if missing_cols:
#                 print(f"[警告] 檔案 {subject_file} 缺少特徵欄位: {missing_cols}，已跳過。")
#                 continue

#             if any(len(subject_data[col].values) == 0 for col in behavioral_features):
#                 print(f"[警告] 檔案 {subject_file} 中部分特徵為空，已跳過。")
#                 continue

#             subject_data_values = np.stack([subject_data[col].values for col in behavioral_features], axis=0)
#             subject_label = labels_df[labels_df['chunk'].str.contains(subject_id)]['label'].values

#             if len(subject_label) > 0:
#                 all_data.append(subject_data_values)
#                 all_labels.append(subject_label[0])
#                 all_filenames.append(filename)
#             else:
#                 print(f"No label found for subject {subject_id}")

#     if len(all_data) == 0:
#         raise ValueError(f"[嚴重警告] 未成功讀取任何有效資料，請檢查資料來源與標籤對應！")

#     all_data = np.array(all_data)
#     all_data = np.expand_dims(all_data, axis=1)
#     all_labels = np.array(all_labels)
#     all_filenames = np.array(all_filenames)
#     return all_data, all_labels, all_filenames

# def remove_nan_samples(data, labels, filenames, name="train"):
#     """
#     剔除含 nan 樣本，並印出被刪除的檔名
#     """
#     data = np.array(data)
#     labels = np.array(labels)
#     filenames = np.array(filenames)
#     nan_mask = np.isnan(data).any(axis=tuple(range(1, data.ndim)))
#     if np.any(nan_mask):
#         print(f"[{name}] 偵測到 {np.sum(nan_mask)} 筆含 nan，將剔除下列檔案：")
#         for fname in filenames[nan_mask]:
#             print(f"    {fname}")
#     keep_mask = ~nan_mask
#     return data[keep_mask], labels[keep_mask], filenames[keep_mask]

# def get_source_data(train_folder_path, test_folder_path, label_file, behavioral_features):
#     """
#     資料自動清除含 nan 並回傳，且印出被刪除檔案
#     """
#     print('\nLoading train data ...')
#     train_data, train_labels, train_filenames = load_csv_data(train_folder_path, label_file, behavioral_features)
#     train_data, train_labels, train_filenames = remove_nan_samples(train_data, train_labels, train_filenames, name="train")

#     train_labels = train_labels.reshape(1, -1)
#     shuffle_index = np.random.permutation(len(train_data))
#     train_data = train_data[shuffle_index, :, :, :]
#     train_labels = train_labels[0][shuffle_index]
#     train_filenames = train_filenames[shuffle_index]

#     print('\nLoading test data ...')
#     test_data, test_labels, test_filenames = load_csv_data(test_folder_path, label_file, behavioral_features)
#     test_data, test_labels, test_filenames = remove_nan_samples(test_data, test_labels, test_filenames, name="test")
#     test_labels = test_labels.reshape(-1)

#     # Standardize
#     target_mean = np.mean(train_data)
#     target_std = np.std(train_data)
#     train_data = (train_data - target_mean) / target_std
#     test_data = (test_data - target_mean) / target_std

#     return train_data, train_labels, test_data, test_labels
