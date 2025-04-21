import os
import re
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

def _extract_label_from_filename(filename):
    """Extracts the class label (0, 1, 2, or 3) from a filename.

    Args:
        filename (str): The base name of the image file.

    Returns:
        int: The extracted class label.

    Raises:
        ValueError: If the label cannot be extracted or is unknown.
    """
    class_label_match = re.search(r'(\d\+|\d)\.(png|jpg|jpeg)', filename)
    if not class_label_match:
        raise ValueError(f"Could not extract class label from filename: {filename}")

    class_str = class_label_match.group(1)
    if class_str == '0':
        return 0
    elif class_str == '1+':
        return 1
    elif class_str == '2+':
        return 2
    elif class_str == '3+':
        return 3
    else:
        raise ValueError(f"Unknown class label string: {class_str} in filename: {filename}")

def get_labels_from_filenames(image_paths):
    """Extracts class labels from a list of image file paths."""
    return [_extract_label_from_filename(os.path.basename(img_path)) for img_path in image_paths]


class BCIDataset(Dataset):
    """Custom Dataset for BCI image classification."""
    def __init__(self, image_paths, transform=None):
        """
        Args:
            image_paths (list): List of paths to the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Loads an image and its corresponding label."""
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = _extract_label_from_filename(os.path.basename(img_path))
        except Exception as e:
            print(f"Error loading or processing image {img_path}: {e}")
            raise

        if self.transform:
            image = self.transform(image)

        return image, label

def _get_image_paths(directory):
    """Recursively finds all image files in a directory."""
    image_paths = []
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) and item.lower().endswith('.png'):
            image_paths.append(item_path)
    return image_paths


def load_dataset(data_dir, batch_size=32, val_split=0.2, num_workers=4):
    """Loads the BCI dataset and prepares DataLoader objects.

    Args:
        data_dir (str): Path to the main dataset directory (containing 'train' and 'test').
        batch_size (int): How many samples per batch to load.
        val_split (float): Fraction of the training data to use for validation.
        num_workers (int): How many subprocesses to use for data loading.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # Optional
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Optional
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Load Training Data ---
    train_dir = os.path.join(data_dir, 'train')
    print(f"Looking for training images in: {train_dir}")
    all_train_images = _get_image_paths(train_dir)
    if not all_train_images:
        raise ValueError(f"No images found in training directory: {train_dir}")
    print(f"Found {len(all_train_images)} total training images.")

    # Split training data into training and validation sets
    try:
        train_labels = get_labels_from_filenames(all_train_images)
        train_imgs, val_imgs = train_test_split(
            all_train_images,
            test_size=val_split,
            random_state=42,
            stratify=train_labels
        )
    except ValueError as e:
        print(f"Error during train/val split: {e}. Stratification might fail if classes are missing.")
        train_imgs, val_imgs = train_test_split(
            all_train_images, test_size=val_split, random_state=42
        )

    print(f"Using {len(train_imgs)} images for training, {len(val_imgs)} for validation.")

    train_dataset = BCIDataset(train_imgs, transform=train_transform)
    val_dataset = BCIDataset(val_imgs, transform=val_test_transform)

    # --- Load Test Data ---
    test_dir = os.path.join(data_dir, 'test')
    print(f"Looking for test images in: {test_dir}")
    test_images = _get_image_paths(test_dir)
    if not test_images:
        print(f"Warning: No images found in test directory: {test_dir}")
        test_dataset = None 
    else:
        print(f"Found {len(test_images)} test images.")
        test_dataset = BCIDataset(test_images, transform=val_test_transform)

    # --- Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        test_loader = None # No test data

    return train_loader, val_loader, test_loader