from dreams_mc.make_model_card import generate_modelcard

config_file_path = './config.yaml'
output_path = './logs/model_card.html'
version_num = '2.0'
generate_modelcard(config_file_path, output_path, version_num)