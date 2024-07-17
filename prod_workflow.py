import mlflow
from pyspark.sql import SparkSession

def create_spark_session():
    spark = SparkSession \
            .builder \
            .appName("LLM ops") \
            .config("spark.llm", "1") \
            .getOrCreate()

    return spark

def read_prod_data(spark, prod_data_path):
    # Spark allows simple scale-out inference for high-throughput, low-cost jobs, and Delta allows us to append to 
    # and modify inference result tables with ACID transactions.  
    prod_data = spark.read.format("delta").load(prod_data_path).limit(10)
    return prod_data

def load_prod_model(spark, prod_model_name):
    # library-agnostic: it never references that the model is a Hugging Face pipeline.*  
    prod_model_udf = mlflow.pyfunc.spark_udf(
        spark,
        model_uri=f"models:/{prod_model_name}/Production",
        env_manager="local",
        result_type="string",
    )

def get_batch_inf(prod_model_udf, prod_data, writetofile = True):
    batch_inference_results = prod_data.withColumn(
        "generated_summary", prod_model_udf("document")
    )

    if writetofile :
        inference_results_path = "inference-results"

        batch_inference_results.write.format("delta").mode("append").save(
            inference_results_path
        )