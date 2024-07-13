from transformers import pipeline

hf_model_name = "t5-small"
min_length = 20
max_length = 40
truncation = True
do_sample = True

summarizer = pipeline(
    task="summarization",
    model=hf_model_name,
    min_length=min_length,
    max_length=max_length,
    truncation=truncation,
    do_sample=do_sample,
    model_kwargs={"cache_dir": DA.paths.datasets},
)  
