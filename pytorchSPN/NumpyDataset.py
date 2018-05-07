import torch
from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    def __init__(self, data_arr, labels=None):
        self.data_array = data_arr
        self.labels = labels
        self.length = len(data_arr)
        self.transforms = None

    def __getitem__(self, index):

        sample = self.data_array[index]
        sample = torch.from_numpy(sample)

        if self.labels is not None:
            label = self.labels[index]
            #label = torch.from_numpy(label)
            return (sample, label)

        return sample

    def __len__(self):
        return self.length
