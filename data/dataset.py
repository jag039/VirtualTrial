import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class TrainGarmentDataset(Dataset):
    def __init__(self, transform=None):
        # Load image paths and categories
        self.image_paths = np.loadtxt('dataset/train.txt', dtype=str)
        self.image_categories = np.loadtxt('dataset/train_cate.txt', dtype=np.int64)
        self.n_samples = len(self.image_paths)
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        category_id = int(self.image_categories[index])

        image = Image.open("dataset/" + image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, category_id

    def __len__(self):
        return self.n_samples


class TestGarmentDataset(Dataset):
    def __init__(self, transform=None):
        # Load image paths and categories
        self.image_paths = np.loadtxt('dataset/test.txt', dtype=str)
        self.image_categories = np.loadtxt('dataset/test_cate.txt', dtype=np.int64)
        self.n_samples = len(self.image_paths)
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        category_id = int(self.image_categories[index])

        image = Image.open("dataset/" + image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, category_id

    def __len__(self):
        return self.n_samples