import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import re
from sklearn.model_selection import train_test_split

class BCIDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Extract class label from the filename
        # Example: 00004_train_1+.png -> 1+ -> 1
        filename = os.path.basename(img_path)
        class_label_match = re.search(r'(\d\+|\d)\.(png|jpg|jpeg)', filename)
        if class_label_match:
            class_str = class_label_match.group(1)
            if class_str == '0':
                label = 0
            elif class_str == '1+':
                label = 1
            elif class_str == '2+':
                label = 2
            elif class_str == '3+':
                label = 3
            else:
                raise ValueError(f"Unknown class label: {class_str}")
        else:
            raise ValueError(f"Could not extract class label from filename: {filename}")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
def load_dataset(data_dir, batch_size=32, val_split=0.2, num_workers=4):
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to a more manageable size for CNN
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Collect all training images
    train_dir = os.path.join(data_dir, 'train')
    print(f"Looking for training images in: {train_dir}")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    
    train_images = [os.path.join(train_dir, img) for img in os.listdir(train_dir) 
                  if img.endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(train_images) == 0:
        raise ValueError(f"No images found in training directory: {train_dir}")
    print(f"Found {len(train_images)} training images")
    
    # Split into train and validation sets
    train_imgs, val_imgs = train_test_split(
        train_images, test_size=val_split, random_state=42, stratify=get_labels_from_filenames(train_images)
    )
    
    # Create datasets
    train_dataset = BCIDataset(train_imgs, transform=train_transform)
    val_dataset = BCIDataset(val_imgs, transform=test_transform)
    
    # Collect all test images
    test_dir = os.path.join(data_dir, 'test')
    print(f"Looking for test images in: {test_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    test_images = [os.path.join(test_dir, img) for img in os.listdir(test_dir) 
                 if img.endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(test_images) == 0:
        raise ValueError(f"No images found in test directory: {test_dir}")
    print(f"Found {len(test_images)} test images")
    
    test_dataset = BCIDataset(test_images, transform=test_transform)
    
    # Create data loaders with specified number of workers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

def get_labels_from_filenames(image_paths):
    """Extract class labels from image filenames for stratified splitting"""
    labels = []
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        class_label_match = re.search(r'(\d\+|\d)\.(png|jpg|jpeg)', filename)
        if class_label_match:
            class_str = class_label_match.group(1)
            if class_str == '0':
                labels.append(0)
            elif class_str == '1+':
                labels.append(1)
            elif class_str == '2+':
                labels.append(2)
            elif class_str == '3+':
                labels.append(3)
        else:
            raise ValueError(f"Could not extract class label from filename: {filename}")
    return labels