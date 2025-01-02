import pandas as pd

import os
save_dir = './logs'
os.makedirs(save_dir, exist_ok=True)

def plot_metrics_table(metrics, save_dir=None):
    df = pd.DataFrame([metrics], columns=["Accuracy", "Precision", "Recall", "F1-Score", "Loss"])
    print(df)

    if save_dir:
        df.to_csv(f'{save_dir}/metrics.csv', index=False)