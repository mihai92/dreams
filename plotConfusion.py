from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

# Create a directory for saving plots, if it doesn't exist
save_dir = './logs'
os.makedirs(save_dir, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, labels, save_dir=None):
    """
    Plots the confusion matrix for a classification task.

    """
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create a heatmap to visualize the confusion matrix
    plt.figure(figsize=(10, 8))  # Set the figure size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)  # Heatmap
    plt.xlabel('Predicted')  # Label for the x-axis
    plt.ylabel('True')  # Label for the y-axis
    plt.title('Confusion Matrix')  # Title of the plot

    # Save the plot image to the specified directory if `save_dir` is provided
    if save_dir:
        plt.savefig(f'{save_dir}/confusion_matrix.png')

    # Display the plot
    plt.show()
