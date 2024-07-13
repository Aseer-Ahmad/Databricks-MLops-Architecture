from transformers import pipeline


def getSummarizationPipeline(config):
 
    # config  = {
    #     "hf_model_name" : "t5-small",
    #     "min_length" : 20,
    #     "max_length" : 40,
    #     "truncation" : True,
    #     "do_sample" : True,
    # }
    
    hf_model_name = config["hf_model_name"]
    min_length = config["min_length"]
    max_length = config["max_length"]
    truncation = config["truncation"]
    do_sample = config["do_sample"]

    summarizer = pipeline(
        task="summarization",
        model=hf_model_name,
        min_length=min_length,
        max_length=max_length,
        truncation=truncation,
        do_sample=do_sample,
        # model_kwargs={"cache_dir": DA.paths.datasets}
    )  

    return summarizer