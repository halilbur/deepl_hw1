import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

def evaluate_model(model, data_loader, device):
    """Evaluate the model on a given dataset"""
    model.eval()
    all_targets = []
    all_predicted = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_targets.extend(targets.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())
    
    accuracy = 100.0 * correct / total
    
    # Classification report
    class_names = ['0', '1+', '2+', '3+']
    report = classification_report(all_targets, all_predicted, target_names=class_names, zero_division=0)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_predicted)
    
    return accuracy, report, conf_matrix

def plot_confusion_matrix(conf_matrix, class_names, save_path):
    """Plots and saves the confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

def visualize_predictions(model, data_loader, device, num_samples=5, save_dir='.'):
    """Visualize some predictions from the data_loader"""
    model.eval()
    try:
        dataiter = iter(data_loader)
        images, labels = next(dataiter)
    except StopIteration:
        print("Data loader is empty, cannot visualize predictions.")
        return
    
    images_dev = images.to(device)
    with torch.no_grad():
        outputs = model(images_dev)
        _, predicted = outputs.max(1)
    
    images = images.cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    plt.figure(figsize=(15, 10))
    actual_samples = min(num_samples, len(images))
    for i in range(actual_samples):
        img = images[i].permute(1, 2, 0)
        img = img * std.permute(1, 2, 0) + mean.permute(1, 2, 0)
        img = img.numpy()
        img = np.clip(img, 0, 1)
        
        plt.subplot(1, actual_samples, i+1)
        plt.imshow(img)
        true_label = labels[i].item()
        pred_label = predicted[i].cpu().item()
        plt.title(f'True: {true_label}, Pred: {pred_label}')
        plt.axis('off')
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'prediction_examples.png')
    plt.savefig(save_path)
    print(f"Prediction examples saved to {save_path}")
    plt.close()