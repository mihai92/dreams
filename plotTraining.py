import matplotlib.pyplot as plt
import os
import yaml
# Create a directory for saving plots, if it doesn't exist
save_dir = './logs'
os.makedirs(save_dir, exist_ok=True)

def plot_training_validation_stats(accuracy_stats, loss_stats, save_dir=None):
    """
    Plots training and validation accuracy and loss over epochs.

    """
    # Unpack statistics
    train_acc = accuracy_stats['train']
    val_acc = accuracy_stats['val']
    train_loss = loss_stats['train']
    val_loss = loss_stats['val']

    # Plot training and validation accuracy
    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.plot(train_acc, label='Train Accuracy')  # Line plot for training accuracy
    plt.plot(val_acc, label='Validation Accuracy')  # Line plot for validation accuracy
    plt.xlabel('Epoch')  # Label for the x-axis
    plt.ylabel('Accuracy')  # Label for the y-axis
    plt.title('Training and Validation Accuracy')  # Title of the plot
    plt.legend()  # Add legend
    if save_dir:
        plt.savefig(f'{save_dir}/training_validation_accuracy.png')  # Save the plot if `save_dir` is provided
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_path = os.path.join(script_dir, "config.yaml")
        with open(config_file_path, "r") as file:
            config_data = yaml.safe_load(file)
        config_data["acc_figpath"]=save_dir+'/training_validation_accuracy.png'
        with open(config_file_path, "w") as file:
            yaml.safe_dump(config_data, file)
        
    plt.show()

    # Plot training and validation loss
    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.plot(train_loss, label='Train Loss')  # Line plot for training loss
    plt.plot(val_loss, label='Validation Loss')  # Line plot for validation loss
    plt.xlabel('Epoch')  # Label for the x-axis
    plt.ylabel('Loss')  # Label for the y-axis
    plt.title('Training and Validation Loss')  # Title of the plot
    plt.legend()  # Add legend
    if save_dir:
        plt.savefig(f'{save_dir}/training_validation_loss.png')  # Save the plot if `save_dir` is provided
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_path = os.path.join(script_dir, "config.yaml")
        with open(config_file_path, "r") as file:
            config_data = yaml.safe_load(file)
        config_data["loss_figpath"]=save_dir+'/training_validation_loss.png'
        with open(config_file_path, "w") as file:
            yaml.safe_dump(config_data, file)
    plt.show()
