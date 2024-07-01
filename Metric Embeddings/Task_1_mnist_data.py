#!/usr/bin/python3

from torch.utils.data import Dataset
from collections import defaultdict
from random import choice
import torchvision
import pdb

class MNISTMetricDataset(Dataset):
    def __init__(self, root="./data", split='train', remove_class=None):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split
        mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=False)
        # self.images, self.targets = mnist_ds.data.float() / 255., mnist_ds.targets
        

        # target2indices is dict like {2: [0, 2, 6, 9], 1: [1, 4, 7], 0: [3, 5, 8]}
        # class 2 has indices [0, 2, 6, 9] in train set
        
        if remove_class is not None:
            # Filter out images with target class equal to remove_class
            indices_to_keep = [i for i in range(len(mnist_ds.targets)) if mnist_ds.targets[i].item() != remove_class]
            self.images = mnist_ds.data[indices_to_keep].float() / 255.
            self.targets = mnist_ds.targets[indices_to_keep]
        else:
            self.images = mnist_ds.data.float() / 255.
            self.targets = mnist_ds.targets

        # self.classes = list(range(10))
        # self.classes = list(range(len(set(self.targets.numpy())))) 
        self.classes = sorted(list(set(self.targets.numpy()))) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # print("Unique Classes:", self.classes)
        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]
        # pdb.set_trace()

    def _sample_negative(self, index):
        anchor_class = self.targets[index].item()
        negative_class = choice([c for c in self.classes if c != anchor_class])
        negative_indices = self.target2indices[negative_class]
        return choice(negative_indices)

    def _sample_positive(self, index):
        anchor_class = self.targets[index].item()               # get the anchor class
        positive_indices = self.target2indices[anchor_class]    # get all the indexes of the anchor class
        positive_indices.remove(index)                          # Remove the anchor itself
        return choice(positive_indices)                         # return a random index from the anchor class

    def __getitem__(self, index):
        anchor = self.images[index].unsqueeze(0)
        target_id = self.targets[index].item()
        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else: # train
            positive = self._sample_positive(index)
            negative = self._sample_negative(index)
            positive = self.images[positive]
            negative = self.images[negative]
            return anchor, positive.unsqueeze(0), negative.unsqueeze(0), target_id

    def __len__(self):
        return len(self.images)
    
    # Returns a sorted list of unique classes in the dataset and their count.
    def get_unique_classes(self):
        return sorted(list(set(self.targets.numpy()))), len(set(self.targets.numpy()))

