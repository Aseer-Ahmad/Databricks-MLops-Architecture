from data import read_yaml, load_dataset
from modelpipeline import getSummarizationPipeline
from experiment import startMLFlowExperiment

def main(YAML_PTH):
    config = read_yaml(YAML_PTH)
    
    data   = load_dataset(config)
    
    model  = getSummarizationPipeline(config)

    EXP_NAME = ''
    startMLFlowExperiment(config, EXP_NAME, model, data)

if __name__ == '__main__':
    YAML_PTH = 'config.yaml'
    main(YAML_PTH)
