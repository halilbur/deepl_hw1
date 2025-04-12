import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from torchvision import transforms

# Define test data transform
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate_model(model, test_loader, device):
    """Evaluate the model on the test set"""
    model.eval()
    all_targets = []
    all_predicted = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
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
    report = classification_report(all_targets, all_predicted, target_names=class_names)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_predicted)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_detailed.png')
    
    return accuracy, report, conf_matrix

def visualize_predictions(model, test_loader, device, num_samples=5):
    """Visualize some predictions from the test set"""
    model.eval()
    # Get a batch from the test loader
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Make predictions
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)
    
    # Convert images for display
    images = images.cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Show images
    plt.figure(figsize=(15, 10))
    for i in range(min(num_samples, len(images))):
        img = images[i].permute(1, 2, 0)  # Change from (C,H,W) to (H,W,C)
        img = img * std.permute(1, 2, 0) + mean.permute(1, 2, 0)  # Denormalize
        img = img.numpy()
        img = np.clip(img, 0, 1)
        
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img)
        plt.title(f'True: {labels[i]}, Pred: {predicted[i].item()}')
        plt.axis('off')
    
    plt.savefig('prediction_examples.png')
    plt.close()