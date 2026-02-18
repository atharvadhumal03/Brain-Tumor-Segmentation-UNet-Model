import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from glob import glob


class BrainMRIDataset(Dataset):
    def __init__(self, data_path, transform=None, patient_list=None):
        """
        Args:
            data_path: Path to the kaggle_3m folder
            transform: Optional transforms to apply
            patient_list: Optional list of patient IDs to include
        """
        self.data_path = data_path
        self.transform = transform
        
        # Get all patient folders
        patient_folders = sorted(glob(os.path.join(data_path, 'TCGA*')))
        
        # Filter by patient_list if provided
        if patient_list is not None:
            patient_folders = [f for f in patient_folders if os.path.basename(f) in patient_list]
        
        # Collect all image-mask pairs
        self.image_paths = []
        self.mask_paths = []
        
        for patient_folder in patient_folders:
            images = sorted(glob(os.path.join(patient_folder, '*[0-9].tif')))
            for img_path in images:
                mask_path = img_path.replace('.tif', '_mask.tif')
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
        
        print(f"Found {len(self.image_paths)} image-mask pairs")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        image = cv2.imread(self.image_paths[idx])
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # type: ignore
        
        # Normalize mask to 0 and 1
        mask = (mask / 255.0).astype(np.float32) # type: ignore
        
        # Apply transforms if any
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask