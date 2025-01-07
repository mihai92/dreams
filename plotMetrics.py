import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml
# Create a directory for saving plots, if it doesn't exist
save_dir = './logs'
os.makedirs(save_dir, exist_ok=True)

def plot_metrics_table(metrics, save_dir=None):
    """
    Saves metrics as a CSV file and prints the metrics in tabular form.

    """
    # Convert metrics to a DataFrame
    df = pd.DataFrame([metrics], columns=["Accuracy", "Precision", "Recall", "F1-Score", "Loss"])
    print(df)  # Print the metrics table

    # Save metrics as a CSV file if `save_dir` is provided
    if save_dir:
        df.to_csv(f'{save_dir}/metrics.csv', index=False)

def plot_metrics_table_as_image(metrics, save_dir=None):
    """
    Saves metrics as an image and displays them in tabular form.

    """
    # Create a DataFrame for the metrics
    df = pd.DataFrame([metrics], columns=["Accuracy", "Precision", "Recall", "F1-Score", "Loss"])

    # Create the figure and axis
    fig, ax = plt.subplots()
    ax.axis('off')  # Turn off the axis

    # Create the table and customize font size and column width
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # Save the figure if a save directory is provided
    if save_dir:
        fig.savefig(f'{save_dir}/metrics_table.png', bbox_inches='tight')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_path = os.path.join(script_dir, "config.yaml")
        with open(config_file_path, "r") as file:
            config_data = yaml.safe_load(file)
        config_data["result_table_figpath"]=save_dir+'/metrics_table.png'
        with open(config_file_path, "w") as file:
            yaml.safe_dump(config_data, file)

    # Show the figure
    plt.show()

    # Explicitly close the figure to avoid memory warnings
    plt.close(fig)
