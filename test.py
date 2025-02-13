import torch
from torch.utils.data import DataLoader
from models.resnet_clothing_model import ClothingClassifier
from data.dataset import TestGarmentDataset
from data.transformation import CustomResNetTransform
from utils.metrics import accuracy
from tqdm import tqdm

NUM_CLASSES = 50

# 1. Prepare the Test Dataset
test_dataset = TestGarmentDataset(transform=CustomResNetTransform())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 2. Initialize the Model and Load the Saved Weights
model = ClothingClassifier(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load('clothing_classifier.pth', map_location=torch.device('cpu')))
model.eval()

# 3. Testing Loop
total_correct = 0
total_samples = 0

with torch.no_grad():
    for images, category_id in tqdm(test_loader, desc="Testing"):
        # Forward pass: get model predictions
        preds = model(images)
        
        # Compute number of correct predictions for this batch
        batch_correct = accuracy(preds, category_id)
        total_correct += batch_correct
        total_samples += images.size(0)

# 4. Compute and Print Test Accuracy
test_accuracy = (total_correct / total_samples) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")
