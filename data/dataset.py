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

        if category_id <= 20:  # Topwear (1-20)
            category_id = int(self.image_categories[index]) - 1
            topwear_presence = 1
            bottomwear_presence = 0
            both_presence = 0
        elif 21 <= category_id <= 36:  # Bottomwear (21-36)
            category_id = int(self.image_categories[index]) - 21
            topwear_presence = 0
            bottomwear_presence = 1
            both_presence = 0
        else:  # Both (37-50)
            category_id = int(self.image_categories[index]) - 37
            topwear_presence = 0
            bottomwear_presence = 0
            both_presence = 1
        
        image = Image.open("dataset/" + image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Convert presence labels to torch.long
        topwear_presence = torch.tensor(topwear_presence, dtype=torch.float32)
        bottomwear_presence = torch.tensor(bottomwear_presence, dtype=torch.float32)
        both_presence = torch.tensor(both_presence, dtype=torch.float32)

        # Return image, presence labels, and category label
        return image, topwear_presence, bottomwear_presence, both_presence, category_id


    def __len__(self):
        return self.n_samples