import mlflow

# Tell MLflow Tracking to use this explicit experiment path,
def startMLFlowExperiment(EXP_NAME, model, data, config):
    mlflow.set_experiment(EXP_NAME)

    with mlflow.start_run():
        
        # LOG PARAMS
        mlflow.log_params(
            {
                "hf_model_name": config["hf_model_name"],
                "min_length": config["min_length"],
                "max_length": config["max_length"],
                "truncation": config["truncation"],
                "do_sample": config["do_sample"],
            }
        )

        # Logged `inputs` are expected to be a list of str, or a list of str->str dicts.
        results = model(data["document"])
        results_list = [r["summary_text"] for r in results]

        # LOG PREDICTIONS
        mlflow.llm.log_predictions(
            inputs=data["document"],
            outputs=results_list,
            prompts=["" for _ in results_list], #coz we dont have any prompts here
        )

        # LOG MODEL SIGNATURE
        # input and output schema for the model.
        signature = mlflow.models.infer_signature(
            data["document"][0],
            mlflow.transformers.generate_signature_output(
                model, data["document"][0]
            ),
        )

        print(f"Signature:\n{signature}\n")

        # LOG INFERENCE CONFIGS
        inference_config = {
            "min_length": config["min_length"],
            "max_length": config["max_length"],
            "truncation": config["truncation"],
            "do_sample": config["do_sample"],
        }

        # Logging a model returns a handle `model_info` to the model metadata in the tracking server.
        # With This `model_info` we can retrieve the logged model.
        model_info = mlflow.transformers.log_model(
            transformers_model=model,
            artifact_path="summarizer",
            task="summarization",
            inference_config=inference_config,
            signature=signature,
            input_example="This is an example of a long news article which this pipeline can summarize for you.",
        )