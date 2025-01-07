from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats
import os
import yaml
# Create a directory for saving plots, if it doesn't exist
save_dir = './logs'
os.makedirs(save_dir, exist_ok=True)

def plot_confidence_intervals(predictions, save_dir=None):
    """
    Plots the 95% confidence intervals for predictions.

    """
    # Calculate the mean prediction across all samples (row-wise average)
    mean_pred = np.mean(predictions, axis=0)

    # Calculate the 95% confidence interval using the standard error of the mean (SEM)
    ci = stats.sem(predictions, axis=0) * stats.t.ppf((1 + 0.95) / 2., len(predictions) - 1)

    # Plot the mean prediction and confidence intervals
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.fill_between(range(len(mean_pred)), mean_pred - ci, mean_pred + ci,
                     color='b', alpha=0.2, label='95% Confidence Interval')  # Shaded region for CI
    plt.plot(mean_pred, color='b', label='Mean Prediction')  # Plot mean prediction as a line
    plt.xlabel('Samples')  # Label for the x-axis
    plt.ylabel('Prediction')  # Label for the y-axis
    plt.title('Confidence Intervals')  # Title of the plot
    plt.legend()  # Add legend to distinguish between mean and CI

    # Save the plot image to the specified directory if `save_dir` is provided
    if save_dir:
        plt.savefig(f'{save_dir}/confidence_intervals.png')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_path = os.path.join(script_dir, "config.yaml")
        with open(config_file_path, "r") as file:
            config_data = yaml.safe_load(file)
        config_data["uncertainty_figpath"]=save_dir+'/confidence_intervals.png'
        with open(config_file_path, "w") as file:
            yaml.safe_dump(config_data, file)

    # Display the plot
    plt.show()
