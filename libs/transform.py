from torchvision import transforms
import torch

class CustomTransforms:
    def __init__(self, mean, std, image_size=224):
        self.mean = mean
        self.std = std
        self.image_size = image_size
        
    def get_eval_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Normalize(self.mean, self.std)
        ])
    
    def get_transform1(self):
        return transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(size=(self.image_size, self.image_size), scale=(0.8, 1)),
            transforms.Normalize(self.mean, self.std)
        ])
    
    def get_transform2(self):
        return transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.Normalize(self.mean, self.std)
        ])

    def get_transform3(self):
        return transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Normalize(self.mean, self.std),
            transforms.RandomErasing(p=1),
        ])

    def get_transform4(self):
        return transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(size=(self.image_size, self.image_size)),
            transforms.Normalize(self.mean, self.std),
            transforms.ElasticTransform(), 
        ])
    
    def get_transform5(self):
        return transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(size=(self.image_size, self.image_size)),
            transforms.Normalize(self.mean, self.std),
        ])
    
    def get_transform6(self):
        return transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(size=(self.image_size, self.image_size)),
            transforms.Normalize(self.mean, self.std),
            transforms.RandomRotation(degrees=(0, 90)),
        ])
    
    def get_all_transforms(self):
        return [
            self.get_transform1(),
            self.get_transform2(),
            self.get_transform3(),
            self.get_transform4(),
            self.get_transform5(),
            self.get_transform6()
        ]