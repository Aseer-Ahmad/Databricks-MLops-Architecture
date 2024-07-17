# Databricks MLOps Architecture

https://mlflow.org/docs/latest/tracking/tracking-api.html

1. data.py <br>
    read and parse config yaml<br>
    read data from hugging face datasets <br>
2. modelpipeline.py
    create a hugging face summarizer pipeline with params
3. experiemnt.py
4. modelregistry.py
5. prod_workflow.py
6. main.py
7. config.yaml
    configuration params for experiments

