from mlflow import MlflowClient
import mlflow


class MLFlowC:
    def __init__(self):
        self.client = MlflowClient()

    def getRegisteredModel(self, model_name):
        self.client.search_registered_models(filter_string=f"name = '{model_name}'")

    def changeStage(self, model_name, model_version = 1, stage ="staging"):
        self.client.transition_model_version_stage(model_name, model_version, stage)

    

def registerModel(REGISTERED_MODEL, model_info):
    # move to the staging step of deployment
    model_name = REGISTERED_MODEL
    model_name = model_name.replace("/", "_").replace(".", "_").replace(":", "_")
    print(model_name)

    # Register a new model under the given name, or a new model version if the name exists already.
    mlflow.register_model(model_uri=model_info.model_uri, name=model_name)



dev_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
dev_model

staging_model = dev_model
