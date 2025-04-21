import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_class_weights(labels, device):
    """Calculate class weights inversely proportional to class frequency"""
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    total_samples = len(labels)
    class_weights = {}
    num_classes = max(class_counts.keys()) + 1
    
    for i in range(num_classes):
        if i in class_counts:
            class_weights[i] = total_samples / (num_classes * class_counts[i])
        else:
            class_weights[i] = 1.0
    
    weights = [class_weights[i] for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float).to(device)

def train_model(model, train_loader, val_loader, device, num_epochs, learning_rate, run_save_dir, run_log_dir): 
    print(f"TensorBoard logs will be saved to: {run_log_dir}")
    
    class_weights = calculate_class_weights([label for _, label in train_loader.dataset], device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class weights for loss function: {class_weights.cpu().numpy()}")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    
    best_loss = float('inf')
    patience = 15
    early_stop_counter = 0
    best_model_path = os.path.join(run_save_dir, 'best_model.pth')
    writer = SummaryWriter(run_log_dir)
    
    # --- Add Model Graph to TensorBoard ---
    # Get a sample batch of inputs from the train_loader
    # try:
    #     dataiter = iter(train_loader)
    #     images, _ = next(dataiter)
    #     images = images.to(device) # Move sample input to the correct device
    #     # Add the model graph
    #     writer.add_graph(model, images)
    #     print("Model graph added to TensorBoard.")
    # except Exception as e:
    #     print(f"Could not add model graph to TensorBoard: {e}")
    # --------------------------------------

    # Store metrics for plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Store predictions and targets for confusion matrix
        all_preds = []
        all_targets = []

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()            
            
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            # Save for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        # Calculate training accuracy and loss for the epoch
        train_accuracy = 100 * train_correct / train_total
        train_loss /= len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        
        # Plot confusion matrix every 10 epochs
        if (epoch + 1) % 10 == 0:
            cm = confusion_matrix(all_targets, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['0', '1+', '2+', '3+'],
                       yticklabels=['0', '1+', '2+', '3+'])
            plt.title(f'Training Confusion Matrix - Epoch {epoch+1}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(run_save_dir, f"train_cm_epoch_{epoch+1}.png")) # Use run_save_dir
            plt.close()

        # Validation step
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        val_accuracy = 100 * val_correct / val_total
        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        # Step the scheduler with validation loss
        scheduler.step(val_loss)
        
        # For final analysis of class-wise performance
        if (epoch + 1) % 10 == 0:
            # Validation confusion matrix
            cm = confusion_matrix(val_targets, val_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['0', '1+', '2+', '3+'],
                       yticklabels=['0', '1+', '2+', '3+'])
            plt.title(f'Validation Confusion Matrix - Epoch {epoch+1}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(run_save_dir, f"val_cm_epoch_{epoch+1}.png")) # Use run_save_dir
            plt.close()
            
            # Print class-wise recall for validation
            print("Validation Class-wise Recall:")
            class_recall = cm.diagonal() / cm.sum(axis=1)
            for i, recall in enumerate(class_recall):
                print(f"Class {i}: {recall:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path) 
            print(f"Saved best model to {best_model_path} at epoch {epoch+1} with validation loss {val_loss:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.")
                break # Stop training

        # Store metrics for plots
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        
        # Log learning rate
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(run_save_dir, "loss_plot.png"))
    plt.close()
    
    # Plot training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(run_save_dir, "accuracy_plot.png"))
    plt.close()

    writer.close() # close the TensorBoard writer
    
    return best_model_path