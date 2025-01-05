from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats
import os
save_dir = './logs'
os.makedirs(save_dir, exist_ok=True)

def plot_class_distribution(dataset, save_dir=None):
    # Count the occurrences of each class
    labels = [label for _, label in dataset]
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Plot the class distribution
    plt.figure(figsize=(8, 6))
    plt.bar(unique_labels, counts, color='skyblue')
    plt.xlabel('Class Labels')
    plt.ylabel('Frequency')
    plt.title('Class Distribution in Dataset')

    if save_dir:
        plt.savefig(f'{save_dir}/data_characteristics.png')
    plt.show()