from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, test_loader, criterion):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in test_loader: #NOSONAR
             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
             outputs = model(X_batch)
             loss = criterion(outputs, y_batch)
             total_loss += loss.item()

             _, preds = torch.max(outputs, 1)
             y_true.extend(y_batch.cpu().numpy())
             y_pred.extend(preds.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return accuracy, precision, recall, f1, total_loss / len(test_loader)