from datasets import load_dataset
from transformers import pipeline


def load_dataset(samples = 10, show = False):

    dataset = load_dataset(
        "xsum", 
        version="1.2.0", 
        # cache_dir=DA.paths.datasets
    ) 

    data = dataset["train"].select(range(samples))

    if show:
        display(data.to_pandas())

    return data