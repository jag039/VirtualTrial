from torchvision import transforms

class CustomResNetTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to 224x224
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet stats
        ])

    def __call__(self, image):
        return self.transform(image)
