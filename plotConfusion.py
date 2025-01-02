from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
save_dir = './logs'
os.makedirs(save_dir, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, labels, save_dir=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    if save_dir:
        plt.savefig(f'{save_dir}/confusion_matrix.png')
    plt.show()