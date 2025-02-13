import torch
from torch.utils.data import DataLoader
from models.resnet_clothing_model import ClothingClassifier
from data.dataset import TrainGarmentDataset
from utils.loss import compute_loss
from utils.metrics import accuracy
from data.transformation import CustomResNetTransform
from tqdm import tqdm

# 1) Prepare Data
train_dataset = TrainGarmentDataset(transform=CustomResNetTransform())
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# 2) Construct loss and optimizer
NUM_CLASSES = 50
model = ClothingClassifier(num_classes=NUM_CLASSES)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3) Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Iterate over training data
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
        for images, category_id in pbar:
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            preds = model(images)
            
            # Compute loss: pass the predictions and target as a dict
            loss = compute_loss(preds, {'category_id': category_id})
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            
            # Update accuracy: accuracy() should return the number of correct predictions for the batch
            batch_correct = accuracy(preds, category_id)
            total_correct += batch_correct
            total_samples += images.size(0)
            
            # Update progress bar
            pbar.set_postfix(loss=loss.item())
    
    # Compute epoch loss and accuracy
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = (total_correct / total_samples) * 100
    print(f"Epoch {epoch+1}/{num_epochs}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.2f}%")

# Optionally, save the model after training:
torch.save(model.state_dict(), 'clothing_classifier.pth')
