import torch
import numpy as np


def interaug(data, label, batch_size, signal_length, device, num_behavioral_features):
    """
    Perform data augmentation by Segmentation and Recombination (S&R) technique.

    Args:
        data (np.array): Data to be augmented.
        label (np.array): Labels corresponding to the input data.
        batch_size (int): Size of the batch.
        signal_length (int): Length of the signal.
        device (torch.device): The device (CPU/GPU) to transfer the tensors.
        num_behavioral_features (int): Number of behavioral features in the data.

    Returns:
        torch.Tensor: Augmented data tensor.
        torch.Tensor: Augmented labels' tensor.
    """

    aug_data, aug_label = [], []
    
        # 防呆：若原始資料或標籤為空，直接回傳零張量
    if data is None or len(data) == 0 or label is None or len(label) == 0:
        print('[Warning] interaug 收到空資料，直接回傳零張量')
        aug_data = torch.zeros((batch_size, 1, num_behavioral_features, signal_length), dtype=torch.float32).to(device)
        aug_label = torch.zeros((batch_size,), dtype=torch.long).to(device)
        return aug_data, aug_label
    
    # Define the number of segments, the length of each segment,
    # the number of augmented samples per class based on the batch size
    n_segments = 30
    segment_length = signal_length // n_segments
    total_samples_per_class = batch_size // len(np.unique(label))

    # Iterate over each unique class
    for cls4aug in np.unique(label):
        # Find indices where current class label exists and extract the corresponding data for this class
        cls_idx = np.where(label == cls4aug)
        tmp_data = data[cls_idx]
        
        # 防呆：該類別資料為空，則略過
        if len(tmp_data) == 0:
            print(f'[Warning] 類別 {cls4aug} 無資料，略過。')
            continue

        # Determine the number of samples to augment to match the total_samples_per_class
        n_samples_needed = total_samples_per_class
        # Initialize a temporary array for augmented data of this class
        tmp_aug_data = np.zeros((n_samples_needed, 1, num_behavioral_features, signal_length))

        # Perform augmentation 
        for ri in range(n_samples_needed):
            for rj in range(n_segments):
                # 若 tmp_data 長度為 1，只能複製自身
                high = max(1, len(tmp_data))
                rand_idx = np.random.randint(0, high, n_segments)
                start = rj * segment_length
                end = (rj + 1) * segment_length
                tmp_aug_data[ri, :, :, start:end] = tmp_data[rand_idx[rj], :, :, start:end]

        aug_data.append(tmp_aug_data)
        aug_label.append(np.full(n_samples_needed, cls4aug))

    # 若所有類別都無資料，回傳全零
    if len(aug_data) == 0:
        print('[Warning] 所有類別均無資料，interaug 回傳零張量')
        aug_data = torch.zeros((batch_size, 1, num_behavioral_features, signal_length), dtype=torch.float32).to(device)
        aug_label = torch.zeros((batch_size,), dtype=torch.long).to(device)
        return aug_data, aug_label

    # Concatenate all augmented data and labels, then shuffle
    aug_data = np.concatenate(aug_data)
    aug_label = np.concatenate(aug_label)
    aug_shuffle = np.random.permutation(len(aug_data))
    aug_data = aug_data[aug_shuffle, :, :]
    aug_label = aug_label[aug_shuffle]

    # Convert numpy arrays to PyTorch tensors and transfer to the specified device
    aug_data = torch.from_numpy(aug_data).to(device).float()
    aug_label = torch.from_numpy(aug_label).to(device).long()

    return aug_data, aug_label


# import torch
# import numpy as np

# def interaug(data, label, batch_size, signal_length, device, num_behavioral_features):
#     """
#     S&R 增強：分段隨機重組。含資料完整性檢查。
#     """
#     aug_data, aug_label = [], []
    
#     if data is None or len(data) == 0 or label is None or len(label) == 0:
#         print('[Warning] interaug 收到空資料，直接回傳零張量')
#         aug_data = torch.zeros((batch_size, 1, num_behavioral_features, signal_length), dtype=torch.float32).to(device)
#         aug_label = torch.zeros((batch_size,), dtype=torch.long).to(device)
#         return aug_data, aug_label

#     n_segments = 30
#     segment_length = signal_length // n_segments
#     unique_classes = np.unique(label)
#     total_samples_per_class = batch_size // len(unique_classes)

#     for cls4aug in unique_classes:
#         cls_idx = np.where(label == cls4aug)
#         tmp_data = data[cls_idx]

#         if len(tmp_data) == 0:
#             print(f'[Warning] 類別 {cls4aug} 無資料，略過。')
#             continue

#         n_samples_needed = total_samples_per_class
#         tmp_aug_data = np.zeros((n_samples_needed, 1, num_behavioral_features, signal_length))

#         for ri in range(n_samples_needed):
#             high = max(1, len(tmp_data))
#             rand_idx = np.random.randint(0, high, n_segments)
#             for rj in range(n_segments):
#                 start = rj * segment_length
#                 end = (rj + 1) * segment_length
#                 if end > signal_length:
#                     end = signal_length
#                 tmp_aug_data[ri, :, :, start:end] = tmp_data[rand_idx[rj], :, :, start:end]

#         aug_data.append(tmp_aug_data)
#         aug_label.append(np.full(n_samples_needed, cls4aug))

#     if len(aug_data) == 0:
#         print('[Warning] 所有類別均無資料，interaug 回傳零張量')
#         aug_data = torch.zeros((batch_size, 1, num_behavioral_features, signal_length), dtype=torch.float32).to(device)
#         aug_label = torch.zeros((batch_size,), dtype=torch.long).to(device)
#         return aug_data, aug_label

#     aug_data = np.concatenate(aug_data)
#     aug_label = np.concatenate(aug_label)
#     aug_shuffle = np.random.permutation(len(aug_data))
#     aug_data = aug_data[aug_shuffle, :, :, :]
#     aug_label = aug_label[aug_shuffle]

#     aug_data = torch.from_numpy(aug_data).to(device).float()
#     aug_label = torch.from_numpy(aug_label).to(device).long()
#     return aug_data, aug_label
