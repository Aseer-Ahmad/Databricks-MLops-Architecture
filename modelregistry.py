from mlflow import MlflowClient
import mlflow


class MLFlowC:
    def __init__(self, model_info):
        self.client = MlflowClient()
        self.model_info = model_info

    def registerModel(self, REGISTERED_MODEL):
        # move to the staging step of deployment
        model_name = REGISTERED_MODEL
        model_name = model_name.replace("/", "_").replace(".", "_").replace(":", "_")
        print(model_name)

        # Register a new model under the given name, or a new model version if the name exists already.
        mlflow.register_model(model_uri=self.model_info.model_uri, name=model_name)

    def getRegisteredModel(self, model_name):
        self.client.search_registered_models(filter_string=f"name = '{model_name}'")

    def queryMLFlowModel(model_info):
        model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
        return model

    def changeStage(self, model_name, model_version = 1, stage ="staging"):
        self.client.transition_model_version_stage(model_name, model_version, stage)

    def predictSample(self,model, data):
        return model.predict(data["document"][0])

    def predictBatch(self, model, data):
        return model.predict(data.to_pandas()["document"])
    



# using common URI patterns for the MLflow Model Registry
dev_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
dev_model

staging_model = dev_model
