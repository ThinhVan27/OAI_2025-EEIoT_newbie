### START: CÁC KHAI BÁO CHÍNH - KHÔNG THAY ĐỔI ###
SEED = 0  # Số seed (Ban tổ chức sẽ công bố & thay đổi vào lúc chấm)
# Đường dẫn đến thư mục train
TEST_DATA_DIR_PATH = 'dataset/test'
# Đường dẫn đến model weights
MODEL_WEIGHTS_PATH = 'output/model_weights.pth'
### END: CÁC KHAI BÁO CHÍNH - KHÔNG THAY ĐỔI ###

### START: CÁC THƯ VIỆN IMPORT ###
import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

### END: CÁC THƯ VIỆN IMPORT ###

### START: SEEDING EVERYTHING - KHÔNG THAY ĐỔI ###
# Seeding nhằm đảm bảo kết quả sẽ cố định
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
### END: SEEDING EVERYTHING - KHÔNG THAY ĐỔI ###

# START: IMPORT CÁC THƯ VIỆN CUSTOM, MODEL, v.v. riêng của nhóm ###
import libs.models as mo
import libs.transform as trf
import libs.predict as pre
### END: IMPORT CÁC THƯ VIỆN CUSTOM, MODEL, v.v. riêng của nhóm ###


### START: CẤU HÌNH CHẠY INFERENCE ###

# Thiết lập device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Định nghĩa class mapping (phải trùng với training)
class_mapping = {
    0: 1,
    1: 2,
    2: 0
}

# Reverse mapping để convert từ model output về class name
reverse_class_mapping = {v: k for k, v in class_mapping.items()}

# Khởi tạo transform cho inference
test_transform = trf.get_test_transforms()

# Kiểm tra đường dẫn dữ liệu test
if not os.path.exists(TEST_DATA_DIR_PATH):
    print(f"⚠️ Error: Test data path not found: {TEST_DATA_DIR_PATH}")
else:
    print(f"✓ Test data path found: {TEST_DATA_DIR_PATH}")

# Kiểm tra model weights
if not os.path.exists(MODEL_WEIGHTS_PATH):
    print(f"⚠️ Error: Model weights not found: {MODEL_WEIGHTS_PATH}")
else:
    print(f"✓ Model weights found: {MODEL_WEIGHTS_PATH}")

### END: CẤU HÌNH CHẠY INFERENCE ###


### START: LOAD MODEL & WEIGHTS ###

# Khởi tạo model
model = mo.efficientnetv2_m_model(num_classes=3, pretrained=False)

# Load weights nếu tồn tại
if os.path.exists(MODEL_WEIGHTS_PATH):
    checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"✓ Model weights loaded successfully from {MODEL_WEIGHTS_PATH}")
else:
    print(f"⚠️ No weights found. Using pretrained weights only.")

model = model.to(device)
model.eval()

### END: LOAD MODEL & WEIGHTS ###


### START: CHUẨN BỊ TEST DATA ###

# Khởi tạo ImageFolder dataset
from torchvision.datasets import ImageFolder

test_dataset = ImageFolder(
    root=TEST_DATA_DIR_PATH,
    transform=test_transform
)

# Tạo DataLoader
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"✓ Test dataset loaded with {len(test_dataset)} images")
print(f"✓ Test loader created with {len(test_loader)} batches")

### END: CHUẨN BỊ TEST DATA ###


### START: CHẠY INFERENCE & TẠO KẾT QUẢ ###

print("\n" + "="*50)
print("STARTING INFERENCE")
print("="*50)

# Gọi hàm predict
pre.predict_and_create_submission_ensemble(
    models=[model],
    test_loader=test_loader,
    class_mapping=class_mapping,
    device=device,
    filename='output/submission.csv'
)

print("\n" + "="*50)
print("INFERENCE COMPLETED")
print("="*50)

# Kiểm tra output file
output_file = 'output/submission.csv'
if os.path.exists(output_file):
    print(f"✓ Submission file created: {output_file}")
    import pandas as pd
    result_df = pd.read_csv(output_file)
    print(f"\nPrediction results preview:")
    print(result_df.head())
    print(f"\nTotal predictions: {len(result_df)}")
else:
    print(f"⚠️ Submission file not found: {output_file}")

### END: CHẠY INFERENCE & TẠO KẾT QUẢ ###
