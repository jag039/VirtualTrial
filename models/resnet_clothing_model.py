import torch
import torch.nn as nn
import torchvision.models as models

class ClothingClassifier(nn.Module):
    def __init__(self, num_topwear_classes, num_bottomwear_classes, num_both_classes):
        super(ClothingClassifier, self).__init__()
        self.num_topwear_classes = num_topwear_classes
        self.num_bottomwear_classes = num_bottomwear_classes
        self.num_both_classes = num_both_classes

        # Loads the pre-trained.resnet model that is allready good at extracting image features
        resnet = models.resnet50(pretrained=True)
        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-1])
        # Resnet feature vector is 2048 dimentions, we add a fully connected layer to drop the dimentionality to 512
        self.fc = nn.Linear(resnet.fc.in_features, 512)

        """
        We need 3 binary classifcation branches to check if there is
        topwear, bottomwear, or both in the photo
        """
        self.presence_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Sigmoid()
        )
        """
        # Separate category classifiers for topwear, bottomwear, and both
        """
        self.topwear_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_topwear_classes)  # Output for topwear categories
        )
        
        self.bottomwear_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_bottomwear_classes)  # Output for bottomwear categories
        )

        self.both_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_both_classes)
        )

    def forward(self, x):
        x = self.resnet_backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        presence_probs = self.presence_classifier(x)
        topwear_category = torch.zeros_like(x)
        bottomwear_category = torch.zeros_like(x)
        both_category = torch.zeros_like(x)

        topwear_presence = presence_probs[:, 0]
        bottomwear_presence = presence_probs[:, 1]
        both_presence = presence_probs[:, 2]

        topwear_category = self.topwear_classifier(x)
        bottomwear_category = self.bottomwear_classifier(x)
        both_category = self.both_classifier(x)

        return topwear_presence, bottomwear_presence, both_presence, topwear_category, bottomwear_category, both_category






    