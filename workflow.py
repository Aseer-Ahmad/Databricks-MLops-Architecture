# Spark allows simple scale-out inference for high-throughput, low-cost jobs, and Delta allows us to append to 
# and modify inference result tables with ACID transactions.  

prod_data = spark.read.format("delta").load(prod_data_path).limit(10)

# Note that the deployment code is library-agnostic: it never references that the model is a Hugging Face pipeline.*  
# This simplified deployment is possible because MLflow logs environment metadata and "knows" how to load the model and run it.

prod_model_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=f"models:/{model_name}/Production",
    env_manager="local",
    result_type="string",
)

batch_inference_results = prod_data.withColumn(
    "generated_summary", prod_model_udf("document")
)
inference_results_path = f"{DA.paths.working_dir}/m6-inference-results".replace(
    "/dbfs", "dbfs:"
)
batch_inference_results.write.format("delta").mode("append").save(
    inference_results_path
)