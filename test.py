from dreams_mc.make_model_card import generate_modelcard
from plotConfidence import plot_confidence_intervals
import numpy as np

#config_file_path = './config.yaml'
#output_path = './logs/model_card.html'
#version_num = '2.0'
#generate_modelcard(config_file_path, output_path, version_num)

predictions = np.array([
    [0.70, 0.20, 0.10],
    [0.15, 0.75, 0.10], 
    [0.05, 0.10, 0.85], 
    [0.40, 0.35, 0.25], 
    [0.60, 0.25, 0.15]
])
plot_confidence_intervals(predictions, save_dir='./logs')