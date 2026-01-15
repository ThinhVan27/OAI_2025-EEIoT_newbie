### START: CÁC KHAI BÁO CHÍNH - KHÔNG THAY ĐỔI ###
SEED = 0  # Số seed (Ban tổ chức sẽ công bố & thay đổi vào lúc chấm)
# Đường dẫn đến thư mục train
# (đúng theo cấu trúc gồm 4 thư mục cho 4 classes của ban tổ chức)
TRAIN_DATA_DIR_PATH = 'data/train'
# Đường dẫn đến thư mục test
TEST_DATA_DIR_PATH = 'data/test'
### END: CÁC KHAI BÁO CHÍNH - KHÔNG THAY ĐỔI ###

### START: CÁC THƯ VIỆN IMPORT ###
# Lưu ý: các thư viện & phiên bản cài đặt vui lòng để trong requirements.txt
import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import transforms
# import tensorflow as tf

### END: CÁC THƯ VIỆN IMPORT ###

### START: SEEDING EVERYTHING - KHÔNG THAY ĐỔI ###
# Seeding nhằm đảm bảo kết quả sẽ cố định
# và không ngẫu nhiên ở các lần chạy khác nhau
# Set seed for random
random.seed(SEED)
# Set seed for numpy
np.random.seed(SEED)
# Set seed for torch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# Set seed for tensorflow
# tf.random.set_seed(SEED)
### END: SEEDING EVERYTHING - KHÔNG THAY ĐỔI ###

# START: IMPORT CÁC THƯ VIỆN CUSTOM, MODEL, v.v. riêng của nhóm ###
import libs.models as mo
import libs.setup_datav2 as sd
import libs.training as tr
import libs.predict as pre
import libs.transform as trf
### END: IMPORT CÁC THƯ VIỆN CUSTOM, MODEL, v.v. riêng của nhóm ###


### START: ĐỊNH NGHĨA & CHẠY HUẤN LUYỆN MÔ HÌNH ###
# Model sẽ được train bằng cac ảnh ở [TRAIN_DATA_DIR_PATH]
class_mapping = {
    0: 1,
    1: 2,
    2: 0
}

############### ROOT DIR ####################
# PLEASE MODIFY IT ADAPTING TO YOUR DIRECTORY (Otherwise, it will be empty)
root_dir = ''

def calculate_mean_std(dataset, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0) 
        images = images.view(batch_samples, 3, -1)  
        mean += images.mean(2).mean(0) 
        std += images.std(2).std(0)  
    mean /= len(loader)
    std /= len(loader)
    return mean, std

mean, std = calculate_mean_std(sd.MushroomDataset(os.path.join(root_dir, TRAIN_DATA_DIR_PATH), 
                                                  transform=transforms.ToTensor()), 
                                                  batch_size=32)
eval_transform = trf.CustomTransforms(mean=mean, std=std, image_size=224).get_eval_transform()
transforms_list = trf.CustomTransforms(mean=mean, std=std, image_size=224).get_all_transforms()

trainloader, valloader, testloader = sd.setup_data_loaders(
    train_dir = os.path.join(root_dir, TRAIN_DATA_DIR_PATH),
    test_dir = os.path.join(root_dir, TEST_DATA_DIR_PATH),
    batch_size=64,
    val_split=0.1,
    transforms_list=transforms_list,
    eval_transform=eval_transform,
    use_multi_augment=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nCreating deep-mutual-learning-models with {3} classes...")

model1 = mo.efficientnetb6_model(num_classes=3).to(device)
model2 = mo.efficientnetv2_m_model(num_classes=3).to(device)
model3 = mo.resnet101_model(num_classes=3).to(device)

print("\n=== Starting model training ===")
trained_model1, trained_model2, trained_model3 = tr.mutual_training_model_v3(
    root_dir=root_dir,
    model1=model1,
    model2=model2,
    model3=model3,
    train_loader=trainloader,
    valid_loader=valloader,
    num_epochs=10,
    learning_rate=0.001,
    device=device
)
print("Training completed!")
### END: ĐỊNH NGHĨA & CHẠY HUẤN LUYỆN MÔ HÌNH ###


### START: THỰC NGHIỆM & XUẤT FILE KẾT QUẢ RA CSV ###
# Kết quả dự đoán của mô hình cho tập dữ liệu các ảnh ở [TEST_DATA_DIR_PATH]
# sẽ lưu vào file "output/results.csv"
# Cấu trúc gồm 2 cột: image_name và label: (encoded: 0, 1, 2, 3)
# image_name,label
# image1.jpg,0
# image2.jpg,1
# image3.jpg,2
# image4.jpg,3
model1.load_state_dict(torch.load(os.path.join(root_dir, 'libs/best_model1.pth')))
model2.load_state_dict(torch.load(os.path.join(root_dir, 'libs/best_model2.pth')))
model3.load_state_dict(torch.load(os.path.join(root_dir, 'libs/best_model3.pth')))

print("Models loaded successfully!")

print("\n=== Making predictions on test data ===")

submission_df = pre.predict_and_create_submission_ensemble(
    models=[model1, model2, model3],
    test_loader=testloader,
    class_mapping=class_mapping,
    device=device,
    filename=os.path.join(root_dir, 'output/results.csv')
)

print("\n=== Process completed successfully! ===")
### END: THỰC NGHIỆM & XUẤT FILE KẾT QUẢ RA CSV ###
