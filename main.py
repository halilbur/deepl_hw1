import os
import argparse
import torch

from dataset import load_dataset
from model import BCICNN
from train import train_model
from evaluate import evaluate_model, visualize_predictions

def main():
    parser = argparse.ArgumentParser(description='Breast Cancer Image Classification with CNN')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save TensorBoard logs')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], 
                        help='Mode: train or test')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print(f"Loading dataset from: {args.data_dir}")

    train_loader, val_loader, test_loader = load_dataset(
        args.data_dir, args.batch_size
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = BCICNN(num_classes=4).to(device)
    
    if args.mode == 'train':
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            save_dir=args.save_dir,
            log_dir=args.log_dir
        )
        
        print("\nEvaluating on test set...")
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))
        accuracy, report, _ = evaluate_model(model, test_loader, device)
        print(f"Test Accuracy: {accuracy:.2f}%")
        print("\nClassification Report:")
        print(report)
        
    elif args.mode == 'test':
        # Load trained model
        model_path = os.path.join(args.save_dir, 'best_model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")
        else:
            print(f"No model found at {model_path}. Please train the model first.")
            return
        
        # Evaluate on test set
        accuracy, report, conf_matrix = evaluate_model(model, test_loader, device)
        print(f"Test Accuracy: {accuracy:.2f}%")
        print("\nClassification Report:")
        print(report)
        
        visualize_predictions(model, test_loader, device, num_samples=5)

if __name__ == "__main__":
    main()