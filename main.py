import os
import argparse
from datetime import datetime
from dataset import load_dataset
from model import BCICNN
from train import train_model
from evaluate import evaluate_model, visualize_predictions, plot_confusion_matrix 
import torch

def main():
    global torch 
    
    parser = argparse.ArgumentParser(description='Breast Cancer Image Classification with CNN')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--save_dir', type=str, default='outputs', help='Directory to save models (train mode) or test results (test mode)') # Clarified help text
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save TensorBoard logs')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], 
                        help='Mode: train or test')
    # Add model_path argument specifically for test mode
    parser.add_argument('--model_path', type=str, help='Path to the model file (.pth) for testing')

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.mode == 'test' and not args.model_path:
        parser.error("--model_path is required when --mode is 'test'")
    if args.mode == 'test' and not os.path.isfile(args.model_path):
         parser.error(f"Model file not found at --model_path: {args.model_path}")
    # -------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = BCICNN(num_classes=4).to(device)

    if args.mode == 'train':
        # --- Create Unique Run Directories for Training --- 
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Include key hyperparameters in the run name
        run_name = f"lr_{args.learning_rate}_bs_{args.batch_size}_{timestamp}"
        # Base save/log directories from args
        base_save_dir = args.save_dir # Now save_dir is the base for outputs
        base_log_dir = args.log_dir
        # Create run-specific subdirectories for models and logs
        run_model_save_dir = os.path.join(base_save_dir, run_name, 'models') # Save models in a subfolder
        run_log_dir = os.path.join(base_log_dir, run_name)
        os.makedirs(run_model_save_dir, exist_ok=True)
        os.makedirs(run_log_dir, exist_ok=True)
        print(f"Run model outputs will be saved to: {run_model_save_dir}")
        print(f"TensorBoard logs for this run: {run_log_dir}")
        # -------------------------------------
        
        print(f"Loading dataset from: {args.data_dir}")
        train_loader, val_loader, test_loader = load_dataset(
            args.data_dir, args.batch_size
        )

        best_model_path = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            run_save_dir=run_model_save_dir, # Pass specific model save dir
            run_log_dir=run_log_dir   # Pass specific log dir
        )
        
        print("\nEvaluating on test set using the best model from this run...")
        if best_model_path and os.path.exists(best_model_path):
            print(f"Loading best model from: {best_model_path}")
            model.load_state_dict(torch.load(best_model_path))
        else:
            if not best_model_path:
                 print("Training did not complete or did not save a best model.")
                 return # Exit if no model was saved
            else: # Path exists but file doesn't - shouldn't happen if saved correctly
                 print(f"Error: File not found at {best_model_path}")
                 return 

        accuracy, report, conf_matrix = evaluate_model(model, test_loader, device)
        print(f"Test Accuracy: {accuracy:.2f}%")
        print("\nClassification Report:")
        print(report)

        # Save evaluation results to the run's main output directory (one level up from models)
        run_output_dir = os.path.dirname(run_model_save_dir) # e.g., outputs/run_name/
        plot_confusion_matrix(conf_matrix, class_names=['0', '1+', '2+', '3+'], 
                              save_path=os.path.join(run_output_dir, 'test_confusion_matrix.png'))
        visualize_predictions(model, test_loader, device, num_samples=5, save_dir=run_output_dir)
        
    elif args.mode == 'test':
        
        # Use args.model_path to load the model
        model_path = args.model_path

        # Determine directory for saving test outputs
        # Default to the directory containing the model if --save_dir is not specified
        if args.save_dir == 'outputs': # Check if default value was used
            test_output_dir = os.path.dirname(model_path)
        else:
            test_output_dir = args.save_dir

        os.makedirs(test_output_dir, exist_ok=True)
        print(f"Test outputs will be saved to: {test_output_dir}")

        print(f"Loading dataset from: {args.data_dir}")

        try:
            # only test_loader is needed in test mode
            _, _, test_loader = load_dataset(
                args.data_dir, args.batch_size
            )
        except ValueError: # If load_dataset only returns one loader in some cases
             print("Assuming load_dataset returned only test_loader.")
             test_loader = load_dataset(args.data_dir, args.batch_size)

        # Load the model from the specified file path
        print(f"Loading model from {model_path}")
        # Load the model state dict onto the correct device
        model.load_state_dict(torch.load(model_path, map_location=device))

        # Evaluate on test set
        accuracy, report, conf_matrix = evaluate_model(model, test_loader, device)
        print(f"Test Accuracy: {accuracy:.2f}%")
        print("\nClassification Report:")
        print(report)
        
        # Save evaluation results to the specified or derived test_output_dir
        plot_confusion_matrix(conf_matrix, class_names=['0', '1+', '2+', '3+'], 
                              save_path=os.path.join(test_output_dir, 'test_confusion_matrix.png'))
        visualize_predictions(model, test_loader, device, num_samples=5, save_dir=test_output_dir)

if __name__ == "__main__":
    main()