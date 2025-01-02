import matplotlib.pyplot as plt
import os
save_dir = './logs'
os.makedirs(save_dir, exist_ok=True)

def plot_training_validation_stats(accuracy_stats, loss_stats, save_dir=None):
    # Unpack statistics
    train_acc = accuracy_stats['train']
    val_acc = accuracy_stats['val']
    train_loss = loss_stats['train']
    val_loss = loss_stats['val']

    # Plot accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Save the plot
    if save_dir:
        plt.savefig(f'{save_dir}/training_validation_stats.png')

    plt.show()