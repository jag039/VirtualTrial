import torch
import torch.nn as nn
import torchvision.models as models

class ClothingClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ClothingClassifier, self).__init__()
        self.num_classes = num_classes

        # Loads the pre-trained.resnet model that is allready good at extracting image features
        resnet = models.resnet50(pretrained=True)
        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-1])
        # Resnet feature vector is 2048 dimentions, we add a fully connected layer to drop the dimentionality to 512
        self.fc = nn.Linear(resnet.fc.in_features, 512)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes) 
        )

    def forward(self, x):
        x = self.resnet_backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        category = self.classifier(x)
        return category    