# Databricks MLOps Architecture

https://mlflow.org/docs/latest/index.html
https://docs.delta.io/latest/quick-start.html#set-up-apache-spark-with-delta-lake

<p align="center">
  <img src="/imgs/learn-core-components.png" width="45%" />
  <img src="/imgs/mlops-lakehouse.png" width="45%" />
</p>


<h2>Delta Format </h2>: <br>
Delta Lake is an open-source storage layer that brings reliability to data lakes. It combines the scalability and cost-effectiveness of data lakes with the reliability and performance of data warehouses. Delta Lake provides ACID (Atomicity, Consistency, Isolation, Durability) transactions, scalable metadata handling, and unifies streaming and batch data processing.

<h2>MLFlow </h2>: <br>
MLflow is an open-source platform designed to streamline the end-to-end machine learning lifecycle. It simplifies various stages, including experimentation, reproducibility, and deployment. MLflow provides four key components: Tracking, Projects, Models, and Model Registry

1. data.py <br>
    - read and parse config yaml<br>
    - read data from hugging face datasets <br>
2. modelpipeline.py <br>
    - create a hugging face summarizer pipeline with params
3. experiemnt.py
    - starts a MLFlow experiment session and sets params to be logged 
4. modelregistry.py
    - contains MLFlow unified API client to control model registry, changing stage, loading reigstered models etc.
5. prod_workflow.py
    - use spark session with delta lake data format for loading production data, model for batch inferences
6. main.py
    - runs an example of experiment logging 
7. config.yaml
    configuration params for experiments

