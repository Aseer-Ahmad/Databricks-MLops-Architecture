loaded_summarizer = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)

loaded_summarizer.predict(xsum_sample["document"][0])

results = loaded_summarizer.predict(xsum_sample.to_pandas()["document"])
