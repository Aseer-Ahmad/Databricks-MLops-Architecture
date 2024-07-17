# Databricks MLOps Architecture

https://mlflow.org/docs/latest/index.html <br>
https://docs.delta.io/latest/quick-start.html#set-up-apache-spark-with-delta-lake

<p align="center">
  <img src="imgs/mlflow lifecycle.png" width="100%" />
</p>

<p align="center">
  <img src="/imgs/mlops-lakehouse.png" width="100%" />
</p>


<h2>Delta Format </h2>
Delta Lake is an open-source storage layer that brings reliability to data lakes. It combines the scalability and cost-effectiveness of data lakes with the reliability and performance of data warehouses. Delta Lake provides ACID (Atomicity, Consistency, Isolation, Durability) transactions, scalable metadata handling, and unifies streaming and batch data processing.

<h2>MLFlow </h2>
MLflow is an open-source platform designed to streamline the end-to-end machine learning lifecycle. It simplifies various stages, including experimentation, reproducibility, and deployment. MLflow provides four key components: Tracking, Projects, Models, and Model Registry

<br>
1. data.py <br>
    - read and parse config yaml
    - read data from hugging face datasets 
2. modelpipeline.py <br>
    - create a hugging face summarizer pipeline with params<br>
3. experiemnt.py<br>
    - starts a MLFlow experiment session and sets params to be logged <br>
4. modelregistry.py<br>
    - contains MLFlow unified API client to control model registry, changing stage, loading reigstered models etc.<br>
5. prod_workflow.py<br>
    - use spark session with delta lake data format for loading production data, model for batch inferences<br>
6. main.py<br>
    - runs an example of experiment logging <br>
7. config.yaml<br>
   - configuration params for experiments<br>

