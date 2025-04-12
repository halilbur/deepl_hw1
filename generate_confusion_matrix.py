import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import multiprocessing

from dataset import load_dataset
from model import BCICNN

def main():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model = BCICNN(num_classes=4).to(device)
    model_path = 'models/best_model.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print("Please make sure you've trained the model first.")
        return
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")

    # Load the test data without multiprocessing
    print("Loading test data...")
    try:
        # Set num_workers=0 to avoid multiprocessing issues
        _, _, test_loader = load_dataset('./BCI_dataset', batch_size=32, num_workers=0)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Collect predictions and true labels
    all_preds = []
    all_labels = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Create confusion matrix
    print("Creating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    class_names = ['0', '1+', '2+', '3+']

    # Calculate accuracy
    correct = np.sum(np.array(all_preds) == np.array(all_labels))
    total = len(all_labels)
    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Print class distribution
    print("\nClass distribution in test set:")
    unique, counts = np.unique(all_labels, return_counts=True)
    for class_idx, count in zip(unique, counts):
        print(f"Class {class_names[class_idx]}: {count} images")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    
    # Save the confusion matrix
    output_file = 'custom_confusion_matrix.png'
    plt.savefig(output_file)
    print(f"\nConfusion matrix saved as '{output_file}'")
    
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot: {e}")
        print("But the image has been saved successfully.")

if __name__ == '__main__':
    # This is necessary for multiprocessing on Windows
    multiprocessing.freeze_support()
    main()