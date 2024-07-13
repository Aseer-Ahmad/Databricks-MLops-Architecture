from mlflow import MlflowClient


# move to the staging step of deployment
model_name = f"summarizer - {DA.username}"
model_name = model_name.replace("/", "_").replace(".", "_").replace(":", "_")
print(model_name)

# Register a new model under the given name, or a new model version if the name exists already.
mlflow.register_model(model_uri=model_info.model_uri, name=model_name)



client = MlflowClient()
client.search_registered_models(filter_string=f"name = '{model_name}'")


model_version = 1
dev_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
dev_model

client.transition_model_version_stage(model_name, model_version, "staging")
staging_model = dev_model
