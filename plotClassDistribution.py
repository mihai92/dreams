from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats
import os
import yaml
# Create a directory for saving plots, if it doesn't exist
save_dir = './logs'
os.makedirs(save_dir, exist_ok=True)

def plot_class_distribution(dataset, save_dir=None):
    """
    Plots the distribution of classes in a given dataset.

    """
    # Extract labels from the dataset
    labels = [label for _, label in dataset]

    # Calculate the frequency of each unique class label
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Create a bar chart to visualize class distribution
    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.bar(unique_labels, counts, color='skyblue')  # Bar chart with class labels and counts
    plt.xlabel('Class Labels')  # Label for the x-axis
    plt.ylabel('Frequency')  # Label for the y-axis
    plt.title('Class Distribution in Dataset')  # Title of the plot

    # Save the plot image to the specified directory if `save_dir` is provided
    if save_dir:
        plt.savefig(f'{save_dir}/data_characteristics.png')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_path = os.path.join(script_dir, "config.yaml")
        with open(config_file_path, "r") as file:
            config_data = yaml.safe_load(file)
        config_data["data_figpath"]=save_dir+'/data_characteristics.png'
        with open(config_file_path, "w") as file:
            yaml.safe_dump(config_data, file)

    # Display the plot
    plt.show()
