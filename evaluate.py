import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.dataset import BrainMRIDataset

# Configuration
DATA_PATH = 'data/mri-segmentation/kaggle_3m/'
MODEL_PATH = 'best_model.pth'
BATCH_SIZE = 16
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Load test dataset
print("Loading test dataset...")
dataset = BrainMRIDataset(data_path=DATA_PATH)
patient_ids = list(set([os.path.basename(os.path.dirname(path)) for path in dataset.image_paths]))

train_patients, test_patients = train_test_split(patient_ids, test_size=0.15, random_state=42)
test_dataset = BrainMRIDataset(DATA_PATH, transform=val_transform, patient_list=test_patients)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Test samples: {len(test_dataset)}")

# Load model
print("Loading model...")
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=3,
    classes=1,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Calculate Dice score
def calculate_dice_score(outputs, masks, threshold=0.5):
    preds = torch.sigmoid(outputs) > threshold
    preds = preds.float()
    
    preds = preds.view(-1)
    masks = masks.view(-1)
    
    intersection = (preds * masks).sum()
    dice = (2. * intersection) / (preds.sum() + masks.sum() + 1e-8)
    
    return dice.item()

# Evaluate on test set
print("\nEvaluating on test set...")
test_dice_scores = []

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE).unsqueeze(1)
        
        outputs = model(images)
        dice = calculate_dice_score(outputs, masks)
        test_dice_scores.append(dice)

avg_test_dice = np.mean(test_dice_scores)
print(f"\nTest Dice Score: {avg_test_dice:.4f}")

# Visualize predictions
print("\nGenerating sample predictions...")
model.eval()

fig, axes = plt.subplots(3, 3, figsize=(12, 12))

with torch.no_grad():
    images, masks = next(iter(test_loader))
    images = images.to(DEVICE)
    outputs = model(images)
    predictions = torch.sigmoid(outputs) > 0.5
    
    for i in range(3):
        # Denormalize image
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        mask = masks[i].cpu().numpy()
        pred = predictions[i].cpu().numpy().squeeze()
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title(f'Prediction')
        axes[i, 2].axis('off')

plt.tight_layout()
plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
print("Saved predictions to predictions.png")
plt.show()

print("\nEvaluation complete!")