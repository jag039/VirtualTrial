import torch
from torch.utils.data import DataLoader
from models.resnet_clothing_model import ClothingClassifier
from data.dataset import TrainGarmentDataset
from utils.loss import compute_loss
from utils.metrics import accuracy
from data.transformation import CustomResNetTransform
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32  # Increase batch size for better training efficiency
train_dataset = TrainGarmentDataset(transform=CustomResNetTransform())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

num_topwear_classes = 20
num_bottomwear_classes = 16
num_both_classes = 14

model = ClothingClassifier(num_topwear_classes=num_topwear_classes,
                           num_bottomwear_classes=num_bottomwear_classes, 
                           num_both_classes=num_both_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3) Training loop
num_epochs = 10  # Number of epochs

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
        for images, topwear_presence, bottomwear_presence, both_presence, category_id in pbar:
            images = images.to(device)
            topwear_presence = topwear_presence.to(device)
            bottomwear_presence = bottomwear_presence.to(device)
            both_presence = both_presence.to(device)
            category_id = category_id.to(device)

            # Forward pass
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
            running_loss += loss.detach().item()
            topwear_preds = torch.round(topwear_p)
            correct_preds += (topwear_preds == topwear_presence).sum().item()
            total_preds += topwear_presence.numel()

            # Update progress bar with loss
            pbar.set_postfix(loss=loss.item(), acc=100 * correct_preds / total_preds)

    avg_loss = running_loss / len(train_loader)
    accuracy = correct_preds / total_preds * 100
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
