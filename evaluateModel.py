from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

# Check if a GPU is available; if not, fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, test_loader, criterion):
    """
    Evaluates the performance of a given model on a test dataset.

    Returns:
    - accuracy: Accuracy of the model on the test set.
    - precision: Precision score (weighted average) on the test set.
    - recall: Recall score (weighted average) on the test set.
    - f1: F1-score (weighted average) on the test set.
    - avg_loss: Average loss over the test dataset.
    """
    
    # Set the model to evaluation mode to disable dropout and batch normalization
    model.eval()
    
    # Initialize lists to store ground truth labels and model predictions
    y_true, y_pred = [], []
    total_loss = 0.0  # Accumulate total loss over all batches

    # Disable gradient computation to speed up inference and save memory
    with torch.no_grad():
        for X_batch, y_batch in test_loader:  # Iterate through the test dataset #NOSONAR
            # Move input and target tensors to the appropriate device (CPU or GPU)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Perform forward pass
            outputs = model(X_batch)
            
            # Calculate the loss for the current batch
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()  # Accumulate loss
            
            # Get predicted labels (class with the highest probability)
            _, preds = torch.max(outputs, 1)
            
            # Append ground truth and predictions to the respective lists
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Compute evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)  # Calculate accuracy
    precision = precision_score(y_true, y_pred, average='weighted')  # Weighted precision
    recall = recall_score(y_true, y_pred, average='weighted')  # Weighted recall
    f1 = f1_score(y_true, y_pred, average='weighted')  # Weighted F1-score
    
    # Return all metrics and the average loss over the test set
    return accuracy, precision, recall, f1, total_loss / len(test_loader)