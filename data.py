from datasets import load_dataset
from transformers import pipeline
import yaml

def read_yaml(yaml_file_path):
    
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)

    print(config)
    
    return config

def load_dataset(config, samples = 10, show = False):

    dataset = load_dataset(
        config['dataset'], 
        version=config['dataset_version'], 
        # cache_dir=DA.paths.datasets
    ) 

    data = dataset["train"].select(range(samples))

    if show:
        display(data.to_pandas())

    return data

if __name__ == '__main__':
    YAML_PTH = 'config.yaml'
    read_yaml(YAML_PTH)