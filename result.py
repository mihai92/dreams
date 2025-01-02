from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import os
import numpy as np

from tqdm import tqdm
from plotConfusion import plot_confusion_matrix
from plotTraining import plot_training_validation_stats
from plotMetrics import plot_metrics_table
from plotConfidence import plot_confidence_intervals
from evaluateModel import evaluate_model

from torch.utils.data import DataLoader, random_split
from torcheeg.datasets import FACEDDataset
from torcheeg import transforms
from torcheeg.datasets.constants import FACED_CHANNEL_LOCATION_DICT
from dreams_mc.make_model_card import generate_modelcard

from torcheeg.models.cnn import TSCeption

# Function to save checkpoints
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    # Save the checkpoint
    torch.save(state, checkpoint_path)

    # If it's the best model, save the best model separately
    if is_best:
        torch.save(state, best_model_path)

data_folder = "./Processed_data"
io_path = ".torcheeg\datasets_1735593565489_9a9Qv"

os.makedirs('./logs', exist_ok=True) #NOSONAR

# Load dataset
dataset = FACEDDataset(root_path=data_folder,
                       io_path=io_path,
                       online_transform=transforms.Compose(
                           [transforms.ToTensor(),
                            transforms.To2d()]),
                       label_transform=transforms.Compose([
                           transforms.Select('valence'),
                           transforms.Lambda(lambda x: x + 1)
                       ]))

# Split dataset into training and validation sets
train_size = 0.8
batch_size = 32
num_train_samples = int(len(dataset) * train_size)
num_val_samples = len(dataset) - num_train_samples

train_dataset, val_dataset = random_split(dataset, [num_train_samples, num_val_samples])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #NOSONAR
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) #NOSONAR

# Initialize the model
model = TSCeption(num_classes=9,
                  num_electrodes=30,
                  sampling_rate=250,
                  num_T=15,
                  num_S=15,
                  hid_channels=32,
                  dropout=0.5)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to compute accuracy
def compute_accuracy(y_pred, y_true):
    _, y_pred_tags = torch.max(y_pred, dim=1)
    correct_pred = (y_pred_tags == y_true).float()
    acc = correct_pred.sum() / len(correct_pred)
    return acc

# Training function
def train(n_epochs, val_acc_max_input, model, optimizer, criterion, scheduler, train_loader, val_loader, checkpoint_path, best_model_path, start_epoch=1):
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

        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

        valid_accuracy = val_epoch_acc / len(val_loader)

        print(f'Epoch {e:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(val_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f} | Val Acc: {val_epoch_acc / len(val_loader):.3f}')

        scheduler.step(val_epoch_loss / len(val_loader))

        checkpoint = {
            'epoch': e + 1,
            'valid_acc_max': valid_accuracy,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

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

# Additional evaluation
results = evaluate_model(model=model, test_loader=train_loader, criterion=criterion)
plot_metrics_table(results, save_dir='./logs')

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

# Generate model card
print("Generating Model Card....")
config_file_path = './config.yaml'
output_path = './logs/model_card.html'
version_num = '1.0'
generate_modelcard(config_file_path, output_path, version_num)
