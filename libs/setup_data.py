import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

class MushroomDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        if not os.path.exists(root_dir):
            print(f"Warning: Directory {root_dir} does not exist.")
            return
        try:
            items = os.listdir(root_dir)
            if self.is_test or any(item.lower().endswith(('.png', '.jpg', '.jpeg')) for item in items):
                self._setup_test_dataset(root_dir)
            else:
                self._setup_train_dataset(root_dir)
        except Exception as e:
            print(f"Error setting up dataset: {e}")
            self.classes = ["unknown"]
            self.class_to_idx = {"unknown": 0}

    def _setup_test_dataset(self, root_dir):
        self.classes = ["unknown"]
        self.class_to_idx = {"unknown": 0}

        for img_name in sorted(os.listdir(root_dir)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root_dir, img_name)
                if os.path.isfile(img_path):
                    self.samples.append((img_path, -1))

    def _setup_train_dataset(self, root_dir):
        self.classes = sorted([d for d in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            placeholder = torch.zeros((3, 32, 32))
            return placeholder, label

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.439673, 0.394798, 0.360261], std=[0.184119, 0.177669, 0.170195]),
])

transform1 = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1)),
    transforms.Normalize(mean=[0.439673, 0.394798, 0.360261], std=[0.184119, 0.177669, 0.170195]),
])

transform2 = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.Normalize(mean=[0.439673, 0.394798, 0.360261], std=[0.184119, 0.177669, 0.170195])
])

transform3 = transforms.Compose([
      transforms.PILToTensor(),
      transforms.ConvertImageDtype(torch.float32),
      transforms.Resize(size=(224, 224)),
      transforms.Normalize(mean=[0.439673, 0.394798, 0.360261], std=[0.184119, 0.177669, 0.170195]),
      transforms.RandomErasing(p=1),
  ])
transform4 = transforms.Compose([
      transforms.PILToTensor(),
      transforms.ConvertImageDtype(torch.float32),
      transforms.Resize(size=(224, 224)),
      transforms.Normalize(mean=[0.439673, 0.394798, 0.360261], std=[0.184119, 0.177669, 0.170195]),
      transforms.ElasticTransform(),
  ])

transform5 = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.439673, 0.394798, 0.360261], std=[0.184119, 0.177669, 0.170195])
])

class MultiAugmentDataset(Dataset):
    def __init__(self, dataset, num_copies=2, transforms_list=None):
        self.dataset = dataset
        self.num_copies = num_copies

        if transforms_list is None:
            self.transforms_list = [dataset.transform] * num_copies
        else:
            assert len(transforms_list) == num_copies, "Number of transforms must match num_copies"
            self.transforms_list = transforms_list

        self.original_transform = dataset.transform

    def __len__(self):
        return len(self.dataset) * self.num_copies

    def __getitem__(self, idx):
        real_idx = idx % len(self.dataset)
        copy_idx = idx // len(self.dataset)

        self.dataset.transform = self.transforms_list[copy_idx]
        image, label = self.dataset[real_idx]
        self.dataset.transform = self.original_transform

        return image, label

def setup_data_loaders(train_dir, test_dir, batch_size=32, val_split=0.2, use_multi_augment=True):
    print("Setting up training and test datasets with mixed augmentation strategies...")

    if not os.path.exists(train_dir):
        print(f"Warning: Training directory {train_dir} does not exist!")
        os.makedirs(train_dir, exist_ok=True)

    train_dataset = MushroomDataset(train_dir, transform=None)

    if use_multi_augment:
        print("Using multi-augmentation strategy (5 copies with different transforms)")
        transforms_list = [transform1, transform2, transform3, transform4, transform5]
        train_dataset = MultiAugmentDataset(
            train_dataset,
            num_copies=5,
            transforms_list=transforms_list
        )
    else:
        train_dataset.transform = transform1

    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, valid_dataset = random_split(
        train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    test_dataset = MushroomDataset(test_dir, transform=eval_transform, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_loader, valid_loader, test_loader
