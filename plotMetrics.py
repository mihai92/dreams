import pandas as pd
import matplotlib.pyplot as plt
import os

save_dir = './logs'
os.makedirs(save_dir, exist_ok=True)

def plot_metrics_table(metrics, save_dir=None):
    # Save metrics as CSV
    df = pd.DataFrame([metrics], columns=["Accuracy", "Precision", "Recall", "F1-Score", "Loss"])
    print(df)

    if save_dir:
        df.to_csv(f'{save_dir}/metrics.csv', index=False)

def plot_metrics_table_as_image(metrics, save_dir=None):
    # Create a DataFrame for the metrics
    df = pd.DataFrame([metrics], columns=["Accuracy", "Precision", "Recall", "F1-Score", "Loss"])
    
    # Create the figure and axis
    fig, ax = plt.subplots()
    ax.axis('off')  # Turn off the axis
    
    # Create the table
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # Save the figure if a save directory is provided
    if save_dir:
        fig.savefig(f'{save_dir}/metrics_table.png', bbox_inches='tight')

    # Show the figure
    plt.show()

    # Explicitly close the figure to avoid memory warnings
    plt.close(fig)
