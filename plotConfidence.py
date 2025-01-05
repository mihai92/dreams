from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats
import os
save_dir = './logs'
os.makedirs(save_dir, exist_ok=True)

def plot_confidence_intervals(predictions, save_dir=None):
    mean_pred = np.mean(predictions, axis=0)
    ci = stats.sem(predictions, axis=0) * stats.t.ppf((1 + 0.95) / 2., len(predictions) - 1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(range(len(mean_pred)), mean_pred - ci, mean_pred + ci, color='b', alpha=0.2, label='95% Confidence Interval')
    plt.plot(mean_pred, color='b', label='Mean Prediction')
    plt.xlabel('Samples')
    plt.ylabel('Prediction')
    plt.title('Confidence Intervals')
    plt.legend()

    if save_dir:
        plt.savefig(f'{save_dir}/confidence_intervals.png')
    plt.show()
    plt.close()