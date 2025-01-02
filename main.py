import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import torch
import torch.nn as nn 
import lightning as l 
from dreams_mc.make_model_card import generate_modelcard # model_card library


config_file_path="D:\\MASTER\\topici\\test\\config.yaml"
output_path= "D:\\MASTER\\topici\\test\\test_card.html"


# Version number of your model
version_num = '1.0'

# Generate the model card
generate_modelcard(config_file_path, output_path, version_num)