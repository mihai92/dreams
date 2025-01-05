from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats
import os

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

    # Display the plot
    plt.show()
