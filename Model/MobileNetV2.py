import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights, EfficientNet_B6_Weights
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import copy
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import ConcatDataset
from google.colab import files
from torchvision.utils import make_grid
import shutil
# =======================================
# SECTION 1: Download and Extract Dataset from Kaggle
# =======================================
def setup_kaggle_and_download():
    """Setup Kaggle credentials and download the dataset."""
    !rm -f /root/.kaggle/kaggle.json

    # Create .kaggle directory if it doesn't exist
    os.makedirs("/root/.kaggle", exist_ok=True)

    # Upload kaggle.json from local machine
    print("Please upload your kaggle.json file...")
    uploaded = files.upload()

    # Move kaggle.json to the proper location
    shutil.move("kaggle.json", "/root/.kaggle/")
    os.chmod("/root/.kaggle/kaggle.json", 600)

    # Download the competition dataset
    print("Downloading dataset from Kaggle...")
    os.system("kaggle competitions download -c aio-hutech")

    # Create data directory and extract dataset
    os.makedirs("data", exist_ok=True)
    print("Extracting dataset...")
    os.system("unzip aio-hutech.zip -d data")

    print("Dataset downloaded and extracted successfully!")

# Class mapping - as in your original code
class_mapping = {
    0: 1,
    1: 3,
    2: 0,
    3: 2
}

# =======================================
# SECTION 2: Data Loading and Augmentation
# =======================================
def explore_directory(path, level=0):
    """Print the directory structure to understand what we're working with"""
    indent = '  ' * level
    print(f"{indent}📁 {os.path.basename(path)}")
    try:
        # List all items in the directory
        items = os.listdir(path)
        for item in items:
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                explore_directory(item_path, level + 1)
            else:
                print(f"{indent}  📄 {item}")
    except Exception as e:
        print(f"{indent}  ❌ Error: {e}")

# Custom Dataset class
class MushroomDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        if not os.path.exists(root_dir):
            print(f"Warning: Directory {root_dir} does not exist. Creating empty dataset.")
            return

        try:
            items = os.listdir(root_dir)
            if is_test or any(item.lower().endswith(('.png', '.jpg', '.jpeg')) for item in items):
                self._setup_test_dataset(root_dir)
            else:
                self._setup_train_dataset(root_dir)
        except Exception as e:
            print(f"Error setting up dataset: {e}")
            self.classes = ["unknown"]
            self.class_to_idx = {"unknown": 0}

    def _setup_test_dataset(self, root_dir):
        """Setup dataset for test directory (direct images)"""
        print(f"Setting up test dataset from {root_dir}")
        self.classes = ["unknown"]
        self.class_to_idx = {"unknown": 0}

        for img_name in os.listdir(root_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only image files
                img_path = os.path.join(root_dir, img_name)
                self.samples.append((img_path, -1))  # -1 as placeholder

    def _setup_train_dataset(self, root_dir):
        """Setup dataset for training directory (with class subdirectories)"""
        print(f"Setting up training dataset from {root_dir}")
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            # Convert label to a tensor
            label = torch.tensor(label, dtype=torch.long)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            placeholder = torch.zeros((3, 224, 224))
            # Still need to convert label to tensor in the error case
            label = torch.tensor(label, dtype=torch.long)
            return placeholder, label

def setup_data_loaders(data_dir="data", batch_size=256, val_split=0.2, target_size=20000):
    """Set up data loaders with spatial augmentation for stable training, avoiding color transformations"""

    # Calculate necessary augmentations
    train_dir = os.path.join(data_dir, 'train')

    # Count original images
    original_image_count = 0
    if os.path.exists(train_dir):
        for class_name in os.listdir(train_dir):
            class_dir = os.path.join(train_dir, class_name)
            if os.path.isdir(class_dir):
                original_image_count += len([f for f in os.listdir(class_dir)
                                           if os.path.isfile(os.path.join(class_dir, f))
                                           and f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"Original image count: {original_image_count}")

    # Calculate augmentations needed
    augmentations_per_image = 100

    # Common normalization parameters for ImageNet
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    transform_sets = [
        # Comprehensive augmentation with multiple spatial transforms
        transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),  # Added vertical flip
            transforms.RandomAffine(
                degrees=(-20, 20),  # Expanded rotation range
                translate=(0.2, 0.2),
                scale=(0.8, 1.2),  # Added scale variation
                shear=(-15, 15)  # More flexible shearing
            ),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.6),  # Increased distortion
            transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.5)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),

        # Zoom and crop with rotation
        transforms.Compose([
            transforms.RandomResizedCrop(
                32,
                scale=(0.6, 1.0),  # Wider scale range
                ratio=(0.75, 1.33)  # More aspect ratio variation
            ),
            transforms.RandomRotation(
                degrees=(-30, 30),  # Expanded rotation
                expand=False  # Maintain original image size
            ),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),

        # Elastic deformation (simulates more complex spatial warping)
        transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ElasticTransform(
                alpha=50.0,  # Deformation intensity
                sigma=5.0    # Smoothness of deformation
            ),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),

        # Extreme perspective and affine transform
        transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomPerspective(
                distortion_scale=0.5,  # More extreme perspective
                p=0.7  # Higher probability
            ),
            transforms.RandomAffine(
                degrees=0,  # No rotation
                translate=(0.1, 0.1),
                scale=(0.7, 1.3),
                shear=(-20, 20)
            ),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),

        # Occlusion and spatial manipulation
        transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
            transforms.RandomErasing(
                p=0.6,  # Higher probability
                scale=(0.02, 0.2),  # Wider occlusion range
                ratio=(0.3, 3.3)    # More varied occlusion shapes
            )
        ]),

        # Combination of rotation and perspective
        transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(
                degrees=(-40, 40),  # Wider rotation
                expand=False,
                center=None,
                fill=0  # Optional: fill color for rotated edges
            ),
            transforms.RandomPerspective(
                distortion_scale=0.3,
                p=0.4
            ),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),

        # Extreme scale and translation
        transforms.Compose([
            transforms.RandomResizedCrop(
                32,
                scale=(0.5, 1.0),  # Very wide scale range
                ratio=(0.5, 2.0)   # Extreme aspect ratios
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.3, 0.3),  # More translation
                scale=(0.6, 1.4)
            ),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),

        # Subtle transformations for minor variations
        transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])
    ]

    # Use only the necessary number of transform sets
    transform_sets = transform_sets[:augmentations_per_image*2]

    # Consistent eval transform
    eval_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    # Check directories
    test_dir = os.path.join(data_dir, 'test')

    # Create datasets with different augmentations
    print("Setting up datasets...")
    if not os.path.exists(train_dir):
        print(f"Warning: Training directory {train_dir} does not exist!")
        os.makedirs(train_dir, exist_ok=True)

    # Create augmented datasets
    train_datasets = []
    for transform in transform_sets:
        train_datasets.append(MushroomDataset(train_dir, transform=transform))

    if all(len(dataset) > 0 for dataset in train_datasets):
        # Calculate class distribution from first dataset
        class_counts = [0] * len(train_datasets[0].classes)
        for _, label in train_datasets[0].samples:
            class_counts[label] += 1
        print(f"Class distribution: {class_counts}")

        # Setup validation data (from first dataset)
        train_size = int((1 - val_split) * len(train_datasets[0]))
        val_size = len(train_datasets[0]) - train_size

        # Use fixed random seed for reproducibility
        generator = torch.Generator().manual_seed(42)
        _, valid_dataset = random_split(
            train_datasets[0],
            [train_size, val_size],
            generator=generator
        )

        # Use validation transform for validation dataset to prevent data leakage
        valid_dataset.dataset.transform = eval_transform

        # Combine all augmented datasets
        from torch.utils.data import ConcatDataset
        combined_train_dataset = ConcatDataset(train_datasets)

        print(f"Augmented dataset size: {len(combined_train_dataset)} images")
    else:
        print("Warning: No training samples found. Using mock data.")
        class MockDataset(Dataset):
            def __init__(self, size=100, num_classes=4):
                self.size = size
                self.classes = ["class1", "class2", "class3", "class4"][:num_classes]
                self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            def __len__(self):
                return self.size
            def __getitem__(self, idx):
                return torch.randn(3, 32, 32), idx % len(self.classes)

        combined_train_dataset = MockDataset(size=target_size)
        valid_dataset = MockDataset(size=int(target_size * val_split))

    # Create test dataset
    test_dataset = None
    if os.path.exists(test_dir):
        # Check if test directory has direct images
        test_has_direct_images = any(
            f.lower().endswith(('.jpg', '.jpeg', '.png'))
            for f in os.listdir(test_dir)
            if os.path.isfile(os.path.join(test_dir, f))
        ) if os.path.exists(test_dir) and os.listdir(test_dir) else True

        test_dataset = MushroomDataset(test_dir, transform=eval_transform, is_test=test_has_direct_images)
    else:
        print(f"Warning: Test directory {test_dir} does not exist. Creating mock test dataset.")
        os.makedirs(test_dir, exist_ok=True)
        class MockTestDataset(Dataset):
            def __init__(self, size=20):
                self.size = size
                self.samples = [(f"mock_image_{i}.jpg", -1) for i in range(size)]
            def __len__(self):
                return self.size
            def __getitem__(self, idx):
                return torch.randn(3, 32, 32), -1
        test_dataset = MockTestDataset()

    # Adjust batch size for stability
    adjusted_batch_size = 256  # Cap batch size to prevent instability

    # Create data loaders
    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=adjusted_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True  # Prevent issues with last incomplete batch
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=adjusted_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=adjusted_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Get class names
    class_names = getattr(train_datasets[0], 'classes', ["class0", "class1", "class2", "class3"])

    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    print(f"Training samples (after augmentation): {len(combined_train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {adjusted_batch_size}")

    return train_loader, valid_loader, test_loader, class_names





def display_representative_images(gan, class_names, num_images=5, figure_size=(15, 10), save_path="representative_gan_images.png"):
    """
    Hiển thị các ảnh đại diện được tạo từ GAN cho mỗi lớp

    Parameters:
    gan (ConditionalGAN): Mô hình GAN đã được huấn luyện
    class_names (list): Danh sách tên các lớp
    num_images (int): Số lượng ảnh hiển thị cho mỗi lớp
    figure_size (tuple): Kích thước của hình ảnh đầu ra
    save_path (str): Đường dẫn để lưu hình ảnh
    """
    import matplotlib.pyplot as plt
    import torch
    import numpy as np

    # Số lượng lớp
    num_classes = len(class_names)

    # Tạo figure với các subplot
    fig, axes = plt.subplots(num_classes, num_images, figsize=figure_size)

    # Nếu chỉ có một lớp, axes sẽ là 1D
    if num_classes == 1:
        axes = [axes]

    # Tạo ảnh cho từng lớp
    for class_idx in range(num_classes):
        # Tạo ảnh với nhiều hạt giống khác nhau để có sự đa dạng
        torch.manual_seed(42 + class_idx)  # Đảm bảo tính tái tạo, nhưng khác nhau giữa các lớp

        # Tạo ảnh bằng cách sử dụng generator cho lớp cụ thể
        with torch.no_grad():
            z = torch.randn(num_images, gan.latent_dim, device=gan.device)
            generated_images = gan.class_generators[class_idx](z)

        # Hiển thị ảnh đã tạo
        for i in range(num_images):
            img = generated_images[i].cpu().clone()

            # Chuyển từ phạm vi [-1, 1] sang [0, 1]
            img = (img + 1) / 2.0
            img = img.clamp(0, 1)

            # Chuyển sang numpy và chuyển đổi từ [C, H, W] sang [H, W, C]
            img = img.permute(1, 2, 0).numpy()

            # Hiển thị ảnh
            if num_classes == 1:
                ax = axes[i]
            else:
                ax = axes[class_idx, i]

            ax.imshow(img)
            ax.axis('off')

            # Thêm tiêu đề cho cột đầu tiên
            if i == 0:
                ax.set_ylabel(f"{class_names[class_idx]}", fontsize=12, rotation=0, labelpad=40,
                             ha='right', va='center')

    # Thêm tiêu đề tổng thể
    plt.suptitle("GAN-Generated Representative Images by Class", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Điều chỉnh để có không gian cho tiêu đề

    # Lưu hình ảnh
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Đã lưu hình ảnh đại diện tại: {save_path}")

    # Hiển thị hình ảnh nếu chạy trong môi trường tương tác
    plt.show()

    return fig

# =======================================
# SECTION: GAN Data Augmentation
# =======================================

class GANDataset(Dataset):
    """Dataset of GAN-generated images"""
    def __init__(self, images, labels, transform=None):
        self.images = images  # Tensor of shape [N, C, H, W]
        self.labels = labels  # Tensor of shape [N]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert from tensor format [-1, 1] to PIL image for transformations
        if isinstance(image, torch.Tensor):
            # Denormalize from [-1, 1] to [0, 1]
            image = (image + 1) / 2.0
            image = image.clamp(0, 1)

            # Convert to PIL
            image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        # Ensure label is a tensor (if it's not already)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)

        return image, label

def visualize_generated_samples(images, labels, class_names, num_samples=10):
    """Visualize a few generated samples for each class"""
    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes, num_samples, figsize=(15, 2*num_classes))

    # Get indices for each class
    for class_idx in range(num_classes):
        class_indices = (labels == class_idx).nonzero(as_tuple=True)[0]

        if len(class_indices) < num_samples:
            print(f"Warning: Only {len(class_indices)} samples available for class {class_idx}")
            selected_indices = class_indices
        else:
            # Randomly select samples
            selected_indices = class_indices[torch.randperm(len(class_indices))[:num_samples]]

        # Plot selected samples
        for i, idx in enumerate(selected_indices):
            if i >= num_samples:
                break

            img = images[idx].clone()

            # Denormalize from [-1, 1] to [0, 1]
            img = (img + 1) / 2.0
            img = img.clamp(0, 1)

            # Convert to numpy and transpose from [C, H, W] to [H, W, C]
            img = img.permute(1, 2, 0).numpy()

            if num_classes == 1:
                ax = axes[i]
            else:
                ax = axes[class_idx, i]

            ax.imshow(img)
            ax.axis('off')

            if i == 0:
                ax.set_title(f"Class: {class_names[class_idx]}", fontsize=8)

    plt.tight_layout()
    plt.savefig('gan_generated_samples.png')
    plt.close()

# =======================================
# SECTION: Improved GAN Architecture
# =======================================

class ImprovedGenerator(nn.Module):
    def __init__(self, latent_dim=100, img_size=32, channels=3):
        super(ImprovedGenerator, self).__init__()
        self.img_size = img_size
        self.init_size = self.img_size // 8  # Smaller initial size for more upsampling stages
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 256 * self.init_size ** 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # More complex and deeper upsampling path
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),

            # First upsampling block
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Second upsampling block
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Third upsampling block
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Final output block with residual connection
            nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class ImprovedDiscriminator(nn.Module):
    def __init__(self, img_size=32, channels=3):
        super(ImprovedDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=False, norm=True):
            block = [
                nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            if norm:
                block.append(nn.BatchNorm2d(out_filters))
            return block

        # More layers for better feature extraction
        self.model = nn.Sequential(
            # First layer without normalization
            *discriminator_block(channels, 32, stride=1, norm=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Calculate the size correctly
        # 5 stride-2 operations reduce dimensions by factor of 2^5 = 32
        # But since the first layer has stride=1, we only reduce by factor of 2^4 = 16
        ds_size = img_size // 16
        self.fc_size = 512 * ds_size * ds_size

        # Use spectral normalization for stability
        self.adv_layer = nn.Sequential(
            nn.Linear(512, 1),  # Only 512 features after adaptive pooling
            nn.Sigmoid()
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)

        # Add debug print to see actual size
        # print(f"Shape after flattening: {out.shape}, Expected: {self.fc_size}")

        validity = self.adv_layer(out)
        return validity


class ImprovedConditionalGAN:
    def __init__(self, latent_dim=128, img_size=32, channels=3, num_classes=4, device=None):
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels
        self.num_classes = num_classes

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Initialize models
        self.generator = ImprovedGenerator(latent_dim, img_size, channels).to(self.device)
        self.discriminator = ImprovedDiscriminator(img_size, channels).to(self.device)

        # Class-specific generators
        self.class_generators = {}
        for class_idx in range(num_classes):
            self.class_generators[class_idx] = ImprovedGenerator(latent_dim, img_size, channels).to(self.device)

        # Use better loss function - BCE with logits for numerical stability
        self.adversarial_loss = nn.BCELoss()

        # Learning rates
        self.lr_g = 0.0002
        self.lr_d = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999

        # Optimizers with better parameters
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=self.lr_g,
            betas=(self.beta1, self.beta2)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr_d,
            betas=(self.beta1, self.beta2)
        )

        # Class-specific optimizers
        self.class_optimizers = {}
        for class_idx in range(num_classes):
            self.class_optimizers[class_idx] = optim.Adam(
                self.class_generators[class_idx].parameters(),
                lr=self.lr_g,
                betas=(self.beta1, self.beta2)
            )

    def train_class_specific_gan(self, dataloader, class_idx, epochs=50, sample_interval=50, n_critic=5):
        """Train a GAN for a specific class with improved training process"""
        print(f"Training GAN for class {class_idx}...")
        generator = self.class_generators[class_idx]
        optimizer_G = self.class_optimizers[class_idx]

        # Extract samples for this class only
        class_samples = []
        for batch_images, batch_labels in dataloader:
            for img, label in zip(batch_images, batch_labels):
                if label == class_idx:
                    class_samples.append(img)

        if len(class_samples) == 0:
            print(f"No samples found for class {class_idx}, skipping GAN training")
            return

        class_samples = torch.stack(class_samples).to(self.device)
        num_samples = len(class_samples)

        print(f"Found {num_samples} training samples for class {class_idx}")

        # Create balanced batches for training
        batch_size = min(64, num_samples)

        # Learning rate schedulers for adaptive learning
        scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, epochs)
        scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, epochs)

        # Keep track of losses
        g_losses = []
        d_losses = []

        # Sample fixed noise to track training progress
        fixed_noise = torch.randn(16, self.latent_dim, device=self.device)

        for epoch in range(epochs):
            # Create random batches
            indices = torch.randperm(num_samples)
            total_g_loss = 0
            total_d_loss = 0
            num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division

            for batch_idx in range(0, num_samples, batch_size):
                batch_indices = indices[batch_idx:min(batch_idx + batch_size, num_samples)]
                real_imgs = class_samples[batch_indices]
                current_batch_size = real_imgs.size(0)

                # Adversarial ground truths with label smoothing for stability
                valid = torch.ones(current_batch_size, 1, device=self.device) * 0.9  # Label smoothing
                fake = torch.zeros(current_batch_size, 1, device=self.device) + 0.1  # Label smoothing

                # ---------------------
                #  Train Discriminator
                # ---------------------
                for _ in range(n_critic):  # Train discriminator more frequently for stability
                    self.optimizer_D.zero_grad()

                    # Sample noise and generate images
                    z = torch.randn(current_batch_size, self.latent_dim, device=self.device)
                    gen_imgs = generator(z)

                    # Detach generated images to avoid backprop through generator
                    fake_validity = self.discriminator(gen_imgs.detach())
                    real_validity = self.discriminator(real_imgs)

                    # Calculate losses
                    fake_loss = self.adversarial_loss(fake_validity, fake)
                    real_loss = self.adversarial_loss(real_validity, valid)
                    d_loss = (real_loss + fake_loss) / 2

                    # Backprop and optimize
                    d_loss.backward()
                    self.optimizer_D.step()

                    total_d_loss += d_loss.item()

                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()

                # Generate new batch of images
                z = torch.randn(current_batch_size, self.latent_dim, device=self.device)
                gen_imgs = generator(z)

                # Measure discriminator's ability to classify generated samples as real
                validity = self.discriminator(gen_imgs)

                # Feature matching loss (helps produce more realistic images)
                g_loss = self.adversarial_loss(validity, valid)

                # Backprop and optimize
                g_loss.backward()
                optimizer_G.step()

                total_g_loss += g_loss.item()

            # Update learning rates
            scheduler_G.step()
            scheduler_D.step()

            # Calculate average losses
            avg_g_loss = total_g_loss / num_batches
            avg_d_loss = total_d_loss / (num_batches * n_critic)
            g_losses.append(avg_g_loss)
            d_losses.append(avg_d_loss)

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"[Class {class_idx}] [Epoch {epoch+1}/{epochs}] "
                    f"[D loss: {avg_d_loss:.4f}] [G loss: {avg_g_loss:.4f}]"
                )

            # Save images at specified intervals
            if (epoch + 1) % sample_interval == 0 or epoch == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    gen_imgs = generator(fixed_noise)
                    self._save_sample_images(gen_imgs, epoch+1, class_idx)

        # Plot loss curves
        self._plot_losses(g_losses, d_losses, class_idx)

        print(f"Finished training GAN for class {class_idx}")

    def _save_sample_images(self, images, epoch, class_idx):
        """Save a grid of generated images"""
        os.makedirs("gan_progress", exist_ok=True)
        save_path = f"gan_progress/class_{class_idx}_epoch_{epoch}.png"

        # Denormalize images
        images_cpu = ((images.cpu() + 1) / 2).clamp(0, 1)

        # Make grid
        grid = torchvision.utils.make_grid(images_cpu, nrow=4, normalize=False)

        # Convert to numpy and transpose
        grid = grid.permute(1, 2, 0).numpy()

        # Save using matplotlib
        plt.figure(figsize=(8, 8))
        plt.imshow(grid)
        plt.axis('off')
        plt.title(f"Class {class_idx} - Epoch {epoch}")
        plt.savefig(save_path)
        plt.close()

    def _plot_losses(self, g_losses, d_losses, class_idx):
        """Plot GAN losses over time"""
        os.makedirs("gan_progress", exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(g_losses, label='Generator')
        plt.plot(d_losses, label='Discriminator')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'GAN Losses - Class {class_idx}')
        plt.savefig(f"gan_progress/class_{class_idx}_losses.png")
        plt.close()

    def train_all_class_gans(self, dataloader, epochs=50, sample_interval=50):
        """Train GANs for all classes"""
        for class_idx in range(self.num_classes):
            self.train_class_specific_gan(dataloader, class_idx, epochs, sample_interval)

    def generate_samples(self, class_idx, num_samples=100):
        """Generate samples for a specific class"""
        generator = self.class_generators[class_idx]
        generator.eval()

        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            gen_imgs = generator(z)

        return gen_imgs

    def save_models(self, directory="gan_models"):
        """Save GAN models"""
        os.makedirs(directory, exist_ok=True)
        for class_idx in range(self.num_classes):
            torch.save(self.class_generators[class_idx].state_dict(),
                     f"{directory}/generator_class_{class_idx}.pth")
        print(f"GAN models saved to {directory}")

    def load_models(self, directory="gan_models"):
        """Load GAN models"""
        for class_idx in range(self.num_classes):
            path = f"{directory}/generator_class_{class_idx}.pth"
            if os.path.exists(path):
                self.class_generators[class_idx].load_state_dict(torch.load(path, map_location=self.device))
                print(f"Loaded GAN model for class {class_idx}")
            else:
                print(f"No saved model found for class {class_idx}")


def generate_improved_gan_augmented_data(train_loader, class_names, transform, gan_epochs=200, samples_per_class=1000):
    """Generate augmented data using improved GAN and add to the original dataset"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize improved GAN with better parameters
    num_classes = len(class_names)
    gan = ImprovedConditionalGAN(
        latent_dim=128,  # Larger latent dimension for more variation
        img_size=32,
        channels=3,
        num_classes=num_classes,
        device=device
    )

    # Check if saved models exist
    gan_model_dir = "gan_models"
    if os.path.exists(gan_model_dir) and len(os.listdir(gan_model_dir)) == num_classes:
        print("Loading pre-trained GAN models...")
        gan.load_models(gan_model_dir)
    else:
        # Train class-specific GANs
        print("Training GAN for data augmentation...")
        gan.train_all_class_gans(train_loader, epochs=gan_epochs, sample_interval=50)
        gan.save_models(gan_model_dir)

    # Generate synthetic samples for each class
    all_generated_images = []
    all_generated_labels = []

    # Quality filter parameters
    quality_threshold = 0.5  # Higher values yield fewer but better images

    for class_idx in range(num_classes):
        print(f"Generating {samples_per_class} samples for class {class_idx}...")

        # Generate more samples than needed to allow filtering
        gen_images = gan.generate_samples(class_idx, num_samples=int(samples_per_class * 1.5))

        # Apply quality filter (examples: variance, discriminator score)
        gen_images_cpu = gen_images.cpu()

        # Calculate variance as a simple quality metric (more variance = more structure)
        image_variance = torch.var(gen_images_cpu.view(gen_images_cpu.size(0), -1), dim=1)

        # Get top X% variance images
        top_indices = torch.argsort(image_variance, descending=True)[:samples_per_class]
        filtered_images = gen_images_cpu[top_indices]

        # Store generated samples with their labels
        all_generated_images.append(filtered_images)
        all_generated_labels.append(torch.full((samples_per_class,), class_idx, dtype=torch.long))

    # Concatenate all generated data
    all_generated_images = torch.cat(all_generated_images, dim=0)
    all_generated_labels = torch.cat(all_generated_labels, dim=0)

    # Create dataset from generated samples
    gan_dataset = GANDataset(all_generated_images, all_generated_labels, transform=transform)

    # Visualize some generated samples
    visualize_generated_samples(all_generated_images, all_generated_labels, class_names, num_samples=10)

    # Generate and save a high-quality visualization
    display_representative_images(gan, class_names, num_images=8,
                               figure_size=(20, 4*len(class_names)),
                               save_path="improved_gan_representative_images.png")

    return gan_dataset


# Update setup_data_loaders function to include improved GAN augmentation
def setup_data_loaders_with_improved_gan(data_dir="data", batch_size=256, val_split=0.2, target_size=20000, use_gan=True, gan_epochs=50):
    """Set up data loaders with improved GAN augmentation"""
    # Get base data loaders first (without GAN)
    train_loader, valid_loader, test_loader, class_names = setup_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=val_split,
        target_size=target_size
    )

    if not use_gan:
        return train_loader, valid_loader, test_loader, class_names

    # Common normalization parameters for ImageNet
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    # Basic transform for GAN-generated images
    basic_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    # Generate augmented data using improved GAN
    gan_dataset = generate_improved_gan_augmented_data(
        train_loader=train_loader,
        class_names=class_names,
        transform=basic_transform,
        gan_epochs=gan_epochs,
        samples_per_class=1000  # More samples per class
    )

    # Combine original training data with GAN-generated data
    original_dataset = train_loader.dataset
    combined_dataset = ConcatDataset([original_dataset, gan_dataset])

    # Create new train loader with combined data
    combined_train_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    print(f"Original training dataset size: {len(original_dataset)}")
    print(f"GAN-generated dataset size: {len(gan_dataset)}")
    print(f"Combined training dataset size: {len(combined_dataset)}")

    return combined_train_loader, valid_loader, test_loader, class_names
# =======================================
# Model Definition and Freezing Layers
# =======================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=12):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ImprovedMushroomClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(ImprovedMushroomClassifier, self).__init__()

        # Sử dụng MobileNetV2 nhưng điều chỉnh cho kích thước ảnh nhỏ hơn
        self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        # Thay đổi lớp đầu tiên để phù hợp với kích thước 32x32 (hoặc lớn hơn nếu bạn thay đổi)
        self.model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)

        # Đóng băng base model trong quá trình training ban đầu
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Xác định số features từ lớp cuối cùng
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 32, 32) # Hoặc kích thước bạn chọn
            features_output = self.model.features(dummy_input)
            num_channels = features_output.shape[1]
            feature_size = features_output.shape[2]
            num_features = features_output.shape[1] * features_output.shape[2] * features_output.shape[3]

        # Thêm attention modules vào sau các features
        self.ca = ChannelAttention(num_channels)
        self.sa = SpatialAttention()

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Thay đổi classifier
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features,256),

            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

        # Khởi tạo weights
        for m in self.model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Trích xuất đặc trưng từ base model
        x = self.model.features(x)
        # Áp dụng attention
        residual = x
        x = x * self.ca(x)  # Apply channel attention
        x = x * self.sa(x)  # Apply spatial attention
        x = x + residual    # Add skip connection

        # Flattening tensor cho classifier
        x = x.view(x.size(0), -1)
        x = self.model.classifier(x)
        return x

    def unfreeze(self, lr_factor=0.1):
        """
        Gradually and carefully unfreeze layers with controlled learning rates
        """
        # Thứ tự giải phóng các layer từ cuối lên
        layers_to_unfreeze = [
            # Mở khóa từng phần của features theo thứ tự
            self.model.features[-5:],  # Các layer cuối cùng của features
            self.model.features[-10:-5],  # Các layer ở giữa
            self.model.features[:-10]  # Các layer đầu tiên
        ]

        # Thiết lập learning rate khác nhau cho từng nhóm layer
        base_lr = 1e-4
        current_params = []

        # Mở khóa từng nhóm layer với learning rate khác nhau
        for i, layer_group in enumerate(layers_to_unfreeze):
            for layer in layer_group:
                for param in layer.parameters():
                    param.requires_grad = True
                    current_params.append(param)

            # Giảm dần learning rate khi mở khóa các layer từ sau ra trước
            group_lr = base_lr * (lr_factor ** (len(layers_to_unfreeze) - i))
            print(f"Unfreezing layer group {i+1} with learning rate: {group_lr}")

        # Mở khóa attention modules với learning rate thấp
        for param in self.ca.parameters():
            param.requires_grad = True
        for param in self.sa.parameters():
            param.requires_grad = True

        # Mở khóa classifier hoàn toàn
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        return


# =======================================
# Training Function with Early Stopping
# =======================================
def train_model(model, train_loader, valid_loader, classses, num_epochs=50, initial_lr=1e-3, weight_decay=5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)

    # Sử dụng CrossEntropyLoss thay vì FocalLoss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Sử dụng AdamW với weight decay tốt hơn
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=5e-4, betas=(0.9, 0.999) )

    # Sử dụng OneCycleLR Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=initial_lr * 2,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        pct_start=0.9,
        div_factor=25,
        final_div_factor=1000,
        anneal_strategy='cos'
    )

    # Tracking
    best_val_acc = 0.0
    patience = 3
    counter = 0
    unfreeze_epoch = num_epochs // 2

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Thêm gradient clipping để ổn định training
    max_grad_norm = 1.0

    print("Starting training...")
    for epoch in range(num_epochs):
        # Unfreeze sau một số epoch nhất định
        if epoch == unfreeze_epoch:
            print("Unfreezing all layers for fine-tuning...")
            model.unfreeze()
            # Điều chỉnh learning rate khi unfreeze
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr / 100.0

        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            # Thêm gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total_samples = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data).item()
                val_total_samples += inputs.size(0)

                # Lưu predictions và labels để tính confusion matrix
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_epoch_loss = val_running_loss / val_total_samples
        val_epoch_acc = val_running_corrects / val_total_samples
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)

        # Hiển thị thông tin chi tiết hơn
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')

        # Tạo confusion matrix mỗi 5 epoch
        if (epoch+1) % 5 == 0 or epoch == num_epochs-1:
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - Epoch {epoch+1}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'confusion_matrix_epoch_{epoch+1}.png')
            plt.close()

        # Save best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), 'best_mushroom_mobilenet_model.pth')
            print(f"✓ Saved best model with validation accuracy: {best_val_acc:.4f}")
            counter = 0  # Reset counter
        else:
            counter += 1

        # Early stopping
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Vẽ đồ thị loss và accuracy
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.axhline(y=0.8, color='r', linestyle='--', label='Target Accuracy (0.8)')
    plt.legend()
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig('improved_mobilenet_training_history.png')
    plt.show()

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    return model


# ===============================================
# Testing and CSV Generation
# ===============================================
def predict_with_tta(model, test_loader, classes, class_mapping, filename='mushroom_mobilenet_submission.csv'):
    """
    Generate predictions with Test Time Augmentation (TTA)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Common normalization parameters for ImageNet
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    # Chuẩn bị multiple augmentations cho TTA
    tta_transforms = [
        # Comprehensive augmentation with multiple spatial transforms
        transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),  # Added vertical flip
            transforms.RandomAffine(
                degrees=(-20, 20),  # Expanded rotation range
                translate=(0.2, 0.2),
                scale=(0.8, 1.2),  # Added scale variation
                shear=(-15, 15)  # More flexible shearing
            ),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.6),  # Increased distortion
            transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.5)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),

        # Zoom and crop with rotation
        transforms.Compose([
            transforms.RandomResizedCrop(
                32,
                scale=(0.6, 1.0),  # Wider scale range
                ratio=(0.75, 1.33)  # More aspect ratio variation
            ),
            transforms.RandomRotation(
                degrees=(-30, 30),  # Expanded rotation
                expand=False  # Maintain original image size
            ),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),

        # Elastic deformation (simulates more complex spatial warping)
        transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ElasticTransform(
                alpha=50.0,  # Deformation intensity
                sigma=5.0    # Smoothness of deformation
            ),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),

        # Extreme perspective and affine transform
        transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomPerspective(
                distortion_scale=0.5,  # More extreme perspective
                p=0.7  # Higher probability
            ),
            transforms.RandomAffine(
                degrees=0,  # No rotation
                translate=(0.1, 0.1),
                scale=(0.7, 1.3),
                shear=(-20, 20)
            ),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),

        # Occlusion and spatial manipulation
        transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
            transforms.RandomErasing(
                p=0.6,  # Higher probability
                scale=(0.02, 0.2),  # Wider occlusion range
                ratio=(0.3, 3.3)    # More varied occlusion shapes
            )
        ]),

        # Combination of rotation and perspective
        transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(
                degrees=(-40, 40),  # Wider rotation
                expand=False,
                center=None,
                fill=0  # Optional: fill color for rotated edges
            ),
            transforms.RandomPerspective(
                distortion_scale=0.3,
                p=0.4
            ),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),

        # Extreme scale and translation
        transforms.Compose([
            transforms.RandomResizedCrop(
                32,
                scale=(0.5, 1.0),  # Very wide scale range
                ratio=(0.5, 2.0)   # Extreme aspect ratios
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.3, 0.3),  # More translation
                scale=(0.6, 1.4)
            ),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),

        # Subtle transformations for minor variations
        transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])
    ]

    all_preds = []
    all_filenames = []
    test_dataset = test_loader.dataset

    # Lấy tất cả filenames
    try:
        for i in range(len(test_dataset)):
            if hasattr(test_dataset, 'samples') and i < len(test_dataset.samples):
                img_path = test_dataset.samples[i][0]
                filename_only = os.path.basename(img_path)
                all_filenames.append(filename_only)
            else:
                all_filenames.append(f"test_image_{i}.jpg")
    except Exception as e:
        print(f"Warning: Couldn't extract filenames: {e}")
        all_filenames = [f"test_image_{i}.jpg" for i in range(len(test_dataset))]

    # Áp dụng TTA cho từng ảnh
    print("Applying Test Time Augmentation...")
    with torch.no_grad():
        for i in range(len(test_dataset)):
            try:
                # Lấy ảnh gốc
                if hasattr(test_dataset, 'samples') and i < len(test_dataset.samples):
                    img_path = test_dataset.samples[i][0]
                    image = Image.open(img_path).convert('RGB')
                else:
                    # Fallback for mock dataset
                    image = None
                    all_preds.append(np.random.randint(0, len(classes)))
                    continue

                # Tích hợp nhiều dự đoán qua TTA
                tta_outputs = []
                for transform in tta_transforms:
                    if image is not None:
                        img_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
                        output = model(img_tensor)
                        tta_outputs.append(output)

                # Kết hợp kết quả từ TTA (soft voting)
                avg_output = torch.mean(torch.stack([out for out in tta_outputs]), dim=0)
                _, pred = torch.max(avg_output, 1)
                all_preds.append(pred.item())

                if (i+1) % 50 == 0:
                    print(f"Processed {i+1}/{len(test_dataset)} test images")

            except Exception as e:
                print(f"Error processing image {i}: {e}")
                # Fallback to random prediction if error occurs
                all_preds.append(np.random.randint(0, len(classes)))

    # Map dự đoán sang class đúng
    all_mapped_preds = []
    for p in all_preds:
        if p in class_mapping:
            all_mapped_preds.append(class_mapping[p])
        else:
            print(f"Warning: Prediction {p} not found in class_mapping.")
            all_mapped_preds.append(p)

    # Format ID column - extract filename without extension
    formatted_ids = [os.path.splitext(fname)[0] for fname in all_filenames]

    # Tạo DataFrame cho submission
    submission_df = pd.DataFrame({
        'id': formatted_ids,
        'label': all_mapped_preds
    })

    # Kiểm tra submission trước khi lưu
    print("\nSubmission sample:")
    print(submission_df.head(10))

    # Lưu file CSV
    submission_df.to_csv(filename, index=False)
    print(f"Submission file created: {filename}")

    return submission_df
# ===============================================
# Main Execution
# ===============================================
# Update the main function to use GAN augmentation
def main_with_gan():
    # Thiết lập seed cho khả năng tái tạo
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Would you like to download the dataset from Kaggle? (yes/no)")
    response = input().lower()
    if response == 'yes' or response == 'y':
        setup_kaggle_and_download()

    # Thiết lập data loaders with GAN augmentation
    print("\nSetting up data loaders with GAN augmentation...")
    train_loader, valid_loader, test_loader, classes = setup_data_loaders_with_improved_gan(
        data_dir="data",
        batch_size=256,
        val_split=0.2,
        target_size=20000,
        use_gan=True,
        gan_epochs=50  # Reduce for faster training, increase for better quality
    )

    # Hiển thị ảnh đại diện từ GAN
    print("\nDisplaying representative GAN images for each class...")
    # Tạo lại GAN để có thể truy cập các generator đã huấn luyện
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gan = ImprovedConditionalGAN(latent_dim=128, img_size=32, channels=3, num_classes=len(classes), device=device)
    gan.load_models("gan_models")  # Load models đã được lưu trong quá trình thiết lập data loader

    # Hiển thị ảnh đại diện cho mỗi lớp
    display_representative_images(gan, classes, num_images=5,
                                 figure_size=(15, 10),
                                 save_path="representative_gan_class_images.png")

    # Tạo mô hình
    num_classes = len(classes)
    print(f"\nCreating improved MobileNetV2 model with {num_classes} classes...")
    model = ImprovedMushroomClassifier(num_classes, dropout_rate=0.3)

    # Train mô hình
    print("\n=== Starting improved model training with GAN-augmented data ===")
    trained_model = train_model(
        model,
        train_loader,
        valid_loader,
        classes,
        num_epochs=50,
        initial_lr=1e-4,
        weight_decay=5e-4
    )
    print("Training completed!")

    # Load mô hình tốt nhất và tạo dự đoán
    print("\n=== Making predictions with TTA on test data ===")

    # Create submission file
    submission_df = predict_with_tta(
        model,
        test_loader,
        classes,
        class_mapping,
        filename='mushroom_gan_mobilenet_submission.csv'
    )

    print("\n=== Process completed successfully! ===")

if __name__ == "__main__":
    main_with_gan()  # Use the GAN-augmented version
    # Download the submission file explicitly
    print("Attempting to download submission file...")
    try:
        files.download('mushroom_gan_mobilenet_submission.csv')
        print("Download initiated. Check your browser's download folder.")
    except Exception as e:
        print(f"Error downloading file: {e}")
        print("Please download the file manually from Colab's file browser.")
