import numpy
import torch.utils.data


class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return numpy.shape(self.dataset)[0]

    def __getitem__(self, index):
        return self.dataset[index]


class LabelledDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return numpy.shape(self.dataset)[0]

    def __getitem__(self, index):
        return self.dataset[index], self.labels[index]
