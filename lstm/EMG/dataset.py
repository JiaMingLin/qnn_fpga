import torch.utils.data as data

import os
import os.path
import numpy as np

class EGMDataset(data.Dataset):
    def __init__(self, path, train = True):
        self.parent_dir = os.path.join(path, 'processed')
        self.train = train
        self.train_val_split = [30, 20]
        self.gesture_dataset = self.read_all_npy()
        self.gesture_dataset = self.flatten(self.gesture_dataset)
    
    def read_all_npy(self):
        g_sample_list = []
        for g_npy in os.listdir(self.parent_dir):
            g_sample_list.append(np.load(os.path.join(self.parent_dir, g_npy)))

        using_data = np.array(g_sample_list)[:, :self.train_val_split[0], :, :]
        if not self.train:
            using_data = np.array(g_sample_list)[:, self.train_val_split[0]: , :, :]
        
        using_data = using_data.astype('float32')
        return using_data
    
    def flatten(self, data):
        data_label_pair_list = []
        train_split_len = data.shape[1]
        for g in range(8):
            label = g
            data_label_pair_list = data_label_pair_list + [(data[g, s_idx, :, :], label) for s_idx in range(train_split_len)]

        return data_label_pair_list

    def __len__(self):
        return len(self.gesture_dataset)

    def __getitem__(self, index):
        (gesture_data, label) = self.gesture_dataset[index]
        gesture_data = np.transpose(gesture_data)
        return gesture_data, label


if __name__ == "__main__":
    ds = EGMDataset(dataset_dir='processed')
    sample, label = ds.__getitem__(4)
    print(sample.shape, label)
    
    

