import mlflow

def queryMLFlowModel(model_info):
    """
    Load a model stored in Python function format
    URI can be either a Git repository URI or a local path
    """
    model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
    return model

def predictSample(model, data):
    return model.predict(data["document"][0])

def predictBatch(model, data):
    return model.predict(data.to_pandas()["document"])
