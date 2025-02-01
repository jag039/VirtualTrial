import torch
from torch.utils.data import DataLoader
from models.resnet_clothing_model import ClothingClassifier
from data.dataset import TrainGarmentDataset
from utils.loss import compute_loss
from utils.metrics import accuracy
from data.dataset import TrainGarmentDataset
from data.transformation import CustomResNetTransform
from tqdm import tqdm

# 1) Prepare Data
train_dataset = TrainGarmentDataset(transform=CustomResNetTransform())
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# 2) Construct loss and optimizer
num_topwear_classes = 20
num_bottomwear_classes = 16
num_both_classes = 14

model = ClothingClassifier( num_topwear_classes=num_topwear_classes,
                            num_bottomwear_classes=num_bottomwear_classes, 
                            num_both_classes=num_both_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3) Training loop
num_epochs = 10  # Number of epochs you want to train for

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
        for images, topwear_presence, bottomwear_presence, both_presence, category_id in pbar:
            # images = images.to(device)
            # topwear_presence = topwear_presence.to(device)
            # bottomwear_presence = bottomwear_presence.to(device)
            # both_presence = both_presence.to(device)
            # category_id = category_id.to(device)
            
            # Forward pass:
            optimizer.zero_grad()
            preds = model(images)
            topwear_p, bottomwear_p, both_p, topwear_c, bottomwear_c, both_c = preds

            # Compute loss
            loss = compute_loss(preds, {
                'category_id': category_id,
                'topwear_presence': topwear_presence,
                'bottomwear_presence': bottomwear_presence,
                'both_presence': both_presence
            })
            
            loss.backward()
            optimizer.step()

            # Track the loss
            running_loss += loss.item()
            topwear_preds = torch.round(preds[0])
            correct_preds += torch.sum(topwear_preds == topwear_presence).item()
            total_preds += topwear_presence.size(0)
    avg_loss = running_loss / len(train_loader)
    accuracy = correct_preds / total_preds * 100
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")