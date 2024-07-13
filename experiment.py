import mlflow

# Tell MLflow Tracking to use this explicit experiment path,
# which is located on the left hand sidebar under Machine Learning -> Experiments 
mlflow.set_experiment(f"/Users/{DA.username}/LLM 06 - MLflow experiment")

with mlflow.start_run():
    # LOG PARAMS
    mlflow.log_params(
        {
            "hf_model_name": hf_model_name,
            "min_length": min_length,
            "max_length": max_length,
            "truncation": truncation,
            "do_sample": do_sample,
        }
    )

    # --------------------------------
    # LOG INPUTS (QUERIES) AND OUTPUTS
    # Logged `inputs` are expected to be a list of str, or a list of str->str dicts.
    results = summarizer(xsum_sample["document"])
    results_list = [r["summary_text"] for r in results]

    # Our LLM pipeline does not have prompts separate from inputs, so we do not log any prompts.
    mlflow.llm.log_predictions(
        inputs=xsum_sample["document"],
        outputs=results_list,
        prompts=["" for _ in results_list],
    )

    # ---------
    # LOG MODEL
    # We next log our LLM pipeline as an MLflow model.
    # This packages the model with useful metadata, such as the library versions used to create it.
    # This metadata makes it much easier to deploy the model downstream.
    # Under the hood, the model format is simply the ML library's native format (Hugging Face for us), plus metadata.

    # It is valuable to log a "signature" with the model telling MLflow the input and output schema for the model.
    signature = mlflow.models.infer_signature(
        xsum_sample["document"][0],
        mlflow.transformers.generate_signature_output(
            summarizer, xsum_sample["document"][0]
        ),
    )
    print(f"Signature:\n{signature}\n")

    # For mlflow.transformers, if there are inference-time configurations,
    # those need to be saved specially in the log_model call (below).
    # This ensures that the pipeline will use these same configurations when re-loaded.
    inference_config = {
        "min_length": min_length,
        "max_length": max_length,
        "truncation": truncation,
        "do_sample": do_sample,
    }

    # Logging a model returns a handle `model_info` to the model metadata in the tracking server.
    # With This `model_info` we can retrieve the logged model.
    model_info = mlflow.transformers.log_model(
        transformers_model=summarizer,
        artifact_path="summarizer",
        task="summarization",
        inference_config=inference_config,
        signature=signature,
        input_example="This is an example of a long news article which this pipeline can summarize for you.",
    )