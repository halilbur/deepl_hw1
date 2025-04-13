import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from losses import FocalLoss

# Calculate class weights based on the training dataset
def calculate_class_weights(train_labels, device):
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(train_labels),
        y=train_labels
    )
    return torch.tensor(class_weights, dtype=torch.float).to(device)

def train_model(model, train_loader, val_loader, device, num_epochs, learning_rate, save_dir, log_dir):    
    # Extract all training labels and calculate class weights
    #train_labels = [label for _, label in train_loader.dataset]  # Get all training labels
    #class_weights = calculate_class_weights(train_labels, device)    
    
    # Manually adjust class weights
    manual_weights = [2.0, 1.5, 1.0, 1.2]  # Increase weight for class 2+
    class_weights = torch.tensor(manual_weights, dtype=torch.float).to(device)
    print(f"Class Weights: {class_weights}")

    # Use class weights in the loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Add learning rate scheduler
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 
    #     mode='min',
    #     factor=0.5,      # Reduce learning rate by half when plateau occurs
    #     patience=5,      # Wait 5 epochs before reducing
    #     min_lr=1e-6,     # Don't go below this learning rate
    #     #verbose=True     # Print when learning rate changes
    # )    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    # Initialize variables for training
    best_loss = float('inf')
    writer = SummaryWriter(log_dir)

    patience = 10
    early_stop_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        # Calculate training accuracy and loss for the epoch
        train_accuracy = 100 * train_correct / train_total
        train_loss /= len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Validation step
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        scheduler.step()

        # Save the best model based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
            print(f"Saved best model at epoch {epoch+1} with validation loss {val_loss:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss")
                break

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

    writer.close()