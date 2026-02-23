import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from src.dataset import BrainMRIDataset

# Configuration
DATA_PATH = 'data/mri-segmentation/kaggle_3m/'
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# Data augmentation transforms
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-15, 15), p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Load dataset and split by patients
print("Loading dataset...")
dataset = BrainMRIDataset(data_path=DATA_PATH)
patient_ids = list(set([os.path.basename(os.path.dirname(path)) for path in dataset.image_paths]))

print(f"Total patients: {len(patient_ids)}")

# Split: 70% train, 15% val, 15% test
train_patients, test_patients = train_test_split(patient_ids, test_size=0.15, random_state=42)
train_patients, val_patients = train_test_split(train_patients, test_size=0.176, random_state=42)

print(f"Train: {len(train_patients)} | Val: {len(val_patients)} | Test: {len(test_patients)}")

# Create datasets with transforms
train_dataset = BrainMRIDataset(DATA_PATH, transform=train_transform, patient_list=train_patients)
val_dataset = BrainMRIDataset(DATA_PATH, transform=val_transform, patient_list=val_patients)
test_dataset = BrainMRIDataset(DATA_PATH, transform=val_transform, patient_list=test_patients)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Initialize model
print("\nInitializing model...")
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights=None, # changed from "imagenet" --> None
    in_channels=3,
    classes=1,
)
model = model.to(DEVICE)

# Loss and optimizer
criterion = smp.losses.DiceLoss(mode='binary')
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Model: U-Net with ResNet50 encoder")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

def calculate_dice_score(outputs, masks, threshold=0.5):
    """Calculate Dice score for batch"""
    preds = torch.sigmoid(outputs) > threshold
    preds = preds.float()
    
    preds = preds.view(-1)
    masks = masks.view(-1)
    
    intersection = (preds * masks).sum()
    dice = (2. * intersection) / (preds.sum() + masks.sum() + 1e-8)
    
    return dice.item()

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        dice = calculate_dice_score(outputs, masks)
        
        running_loss += loss.item()
        running_dice += dice
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    
    return epoch_loss, epoch_dice

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    pbar = tqdm(dataloader, desc='Validation')
    
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            dice = calculate_dice_score(outputs, masks)

            running_loss += loss.item()
            running_dice += dice
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})
    
    val_loss = running_loss / len(dataloader)
    val_dice = running_dice / len(dataloader)
    
    return val_loss, val_dice


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    
    train_losses, val_losses = [], []
    train_dices, val_dices = [], []
    
    best_val_dice = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        train_dices.append(train_dice)
        
        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        
        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        
        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ Saved best model (Dice: {val_dice:.4f})")
    
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best Val Dice Score: {best_val_dice:.4f}")
    print("="*50)