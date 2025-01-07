# Import necessary libraries for data handling, model training, evaluation, and visualization
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import os
import numpy as np

# Progress bar for training visualization
from tqdm import tqdm

# Custom plotting and evaluation utilities
from plotConfusion import plot_confusion_matrix
from plotTraining import plot_training_validation_stats
from plotMetrics import plot_metrics_table, plot_metrics_table_as_image
from plotConfidence import plot_confidence_intervals
from evaluateModel import evaluate_model
from plotClassDistribution import plot_class_distribution

# EEG dataset and transformations from TorchEEG
from torch.utils.data import DataLoader, random_split
from torcheeg.datasets import FACEDDataset
from torcheeg import transforms
from torcheeg.datasets.constants import FACED_CHANNEL_LOCATION_DICT

# Utility to generate model cards


# Model architecture for EEG-based classification
from torcheeg.models.cnn import TSCeption

# Function to save checkpoints during training
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    Save model checkpoint.

    Parameters:
    - state: Dictionary containing model state and optimizer state.
    - is_best: Boolean, whether the model is the best so far.
    - checkpoint_path: Path to save the current checkpoint.
    - best_model_path: Path to save the best model checkpoint.
    """
    torch.save(state, checkpoint_path)

    # Save as the best model if applicable
    if is_best:
        torch.save(state, best_model_path)

# Directory for preprocessed data and model logs
data_folder = "./Processed_data"
io_path = ".torcheeg/datasets_1736033063491_oNi60"
os.makedirs('./logs', exist_ok=True)  # Create logs directory if not exists #NOSONAR

# Load the FACED dataset for emotion classification
dataset = FACEDDataset(
    root_path=data_folder,
    io_path=io_path,
    online_transform=transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.To2d()       # Reshape to 2D
    ]),
    label_transform=transforms.Compose([
        transforms.Select('valence'),  # Select the 'valence' label
        transforms.Lambda(lambda x: x + 1)  # Shift labels by 1
    ])
)

# Split dataset into training and validation sets
train_size = 0.8
batch_size = 32
num_train_samples = int(len(dataset) * train_size)
num_val_samples = len(dataset) - num_train_samples

train_dataset, val_dataset = random_split(dataset, [num_train_samples, num_val_samples])

# Dataloader for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #NOSONAR
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) #NOSONAR

# Initialize the EEG classification model
model = TSCeption(
    num_classes=9,          # Number of output classes
    num_electrodes=30,      # Number of input electrodes
    sampling_rate=250,      # Sampling rate of EEG signals
    num_T=15,               # Number of temporal filters
    num_S=15,               # Number of spatial filters
    hid_channels=32,        # Number of hidden channels
    dropout=0.5             # Dropout rate
)

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to compute accuracy for model evaluation
def compute_accuracy(y_pred, y_true):
    """
    Compute accuracy given model predictions and ground truth labels.

    Parameters:
    - y_pred: Tensor of predicted logits or probabilities.
    - y_true: Tensor of true class labels.

    Returns:
    - Accuracy as a floating-point value.
    """
    _, y_pred_tags = torch.max(y_pred, dim=1)  # Get predicted class indices
    correct_pred = (y_pred_tags == y_true).float()  # Count correct predictions
    acc = correct_pred.sum() / len(correct_pred)  # Calculate accuracy
    return acc

# Function to train the model
def train(n_epochs, val_acc_max_input, model, optimizer, criterion, scheduler, train_loader, val_loader, checkpoint_path, best_model_path, start_epoch=1):
    """
    Train the model over multiple epochs with validation.

    Parameters:
    - n_epochs: Number of training epochs.
    - val_acc_max_input: Initial maximum validation accuracy.
    - model: PyTorch model to train.
    - optimizer: Optimizer for weight updates.
    - criterion: Loss function.
    - scheduler: Learning rate scheduler.
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - checkpoint_path: Path to save model checkpoints.
    - best_model_path: Path to save the best model checkpoint.
    - start_epoch: Starting epoch for training.

    Returns:
    - Trained model, accuracy statistics, and loss statistics.
    """
    val_acc_max = val_acc_max_input
    accuracy_stats = {'train': [], 'val': []}
    loss_stats = {'train': [], 'val': []}

    for e in tqdm(range(start_epoch, n_epochs + 1), desc="Training Progress"):
        # Training phase
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader: #NOSONAR
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch)
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = compute_accuracy(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        # Validation phase
        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0
            model.eval()
            for X_val_batch, y_val_batch in val_loader: #NOSONAR
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch)
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = compute_accuracy(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        # Log training and validation statistics
        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

        valid_accuracy = val_epoch_acc / len(val_loader)

        print(f'Epoch {e:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(val_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f} | Val Acc: {val_epoch_acc / len(val_loader):.3f}')

        # Adjust learning rate based on validation loss
        scheduler.step(val_epoch_loss / len(val_loader))

        # Save current checkpoint
        checkpoint = {
            'epoch': e + 1,
            'valid_acc_max': valid_accuracy,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        # Save best model checkpoint
        if valid_accuracy > val_acc_max:
            print(f'Validation accuracy increased ({val_acc_max:.6f} --> {valid_accuracy:.6f}).  Saving model ...')
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            val_acc_max = valid_accuracy

    return model, accuracy_stats, loss_stats


# Class names
class_names = ['Negative', 'Neutral', 'Positive']
model = model.to(device)

# Optimizer, scheduler, and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
criterion = nn.CrossEntropyLoss()

# Training the model
valid_acc_max = 0.0
trained_model, accuracy_stats, loss_stats = train(220, valid_acc_max, model, optimizer, criterion, scheduler, train_loader, val_loader, "./logs/current_checkpoint.pt", "./logs/best_model.pt", start_epoch=1)

# Plot training and validation stats
plot_training_validation_stats(accuracy_stats, loss_stats, save_dir='./logs')

# Load the best model for evaluation
checkpoint_path = './logs/best_model.pt'
checkpoint = torch.load(checkpoint_path, map_location=device)
print("Checkpoint keys:", checkpoint.keys())
model.load_state_dict(checkpoint['state_dict'])
model.to(device)

# Evaluate the model
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, preds = torch.max(outputs, dim=1)

        y_true.extend(y_batch.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Plot confusion matrix
plot_confusion_matrix(y_true=y_true, y_pred=y_pred, labels=class_names, save_dir='./logs')

# Plot data distribution
plot_class_distribution(dataset, save_dir='./logs')

# Additional evaluation
results = evaluate_model(model=model, test_loader=train_loader, criterion=criterion)
plot_metrics_table(results, save_dir='./logs')
plot_metrics_table_as_image(results, save_dir='./logs')

# Collect predictions and their probabilities
predictions = []
with torch.no_grad():
    for X_batch, _ in val_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        predictions.extend(torch.softmax(outputs, dim=1).cpu().numpy())

# Convert to NumPy array
predictions = np.array(predictions)

# Call plot_confidence_intervals with correct predictions
plot_confidence_intervals(predictions=predictions, save_dir='./logs')



