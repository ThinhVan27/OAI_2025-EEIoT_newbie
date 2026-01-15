import os
import torch
import torch.nn as nn
from torchvision import models
import pandas as pd

def predict_and_create_submission_ensemble(models, test_loader, class_mapping, device, filename='submission.csv'):

    for i, model in enumerate(models):
        models[i] = model.to(device)
        models[i].eval()
        print(f"Model {i+1} running on: {next(models[i].parameters()).device}")

    all_preds = []
    all_mapped_preds = []
    all_filenames = []

    try:
        test_dataset = test_loader.dataset
        for i in range(len(test_dataset)):
            if hasattr(test_dataset, 'samples') and i < len(test_dataset.samples):
                img_path = test_dataset.samples[i][0]
                filename_only = os.path.basename(img_path)
                all_filenames.append(filename_only)
            else:
                all_filenames.append(f"test_image_{i}.jpg")
    except Exception as e:
        print(f"Warning: Couldn't extract filenames from test dataset: {e}")
        all_filenames = [f"test_image_{i}.jpg" for i in range(len(test_loader.dataset))]

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)

            batch_preds = []
            for model in models:
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                batch_preds.append(probs)

            ensemble_probs = sum(batch_preds) / len(models)
            _, preds = torch.max(ensemble_probs, 1)

            all_preds.extend(preds.cpu().numpy())

            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx}/{len(test_loader)} batches")

    for p in all_preds:
        if p in class_mapping:
            all_mapped_preds.append(class_mapping[p])
        else:
            print(f"Warning: Prediction {p} not found in class_mapping. Using raw prediction.")
            all_mapped_preds.append(p)

    formatted_ids = [os.path.splitext(fname)[0] for fname in all_filenames]

    submission_df = pd.DataFrame({
        'image_name': formatted_ids,
        'label': all_mapped_preds
    })

    print("\nSubmission sample (BEFORE saving):")
    print(submission_df.head(10))

    submission_df.to_csv(filename, index=False)
    print(f"Submission file created: {filename}")

    print("\nVerifying saved file by reading it back:")
    try:
        saved_df = pd.read_csv(filename)
        print("Sample from saved file:")
        print(saved_df.head(10))

        if not submission_df.equals(saved_df):
            print("WARNING: The saved file differs from the generated predictions!")
            diff_mask = submission_df != saved_df
            diff_indices = diff_mask.any(axis=1)
            if diff_indices.any():
                print("Differences found at rows:")
                print(submission_df[diff_indices].compare(saved_df[diff_indices]))
    except Exception as e:
        print(f"Error verifying saved file: {e}")

    return submission_df