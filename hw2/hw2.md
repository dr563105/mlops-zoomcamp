# Homework week 2 on Experiment Tracking using MLflow

## Question 1:
Version of mlflow

```bash
mlflow --version
mlflow, version 1.26.1
```
### Answer: v1.26.1 

## Question 2:
Number of files in the output folder?

```bash
cd output/
ls
dv.pkl  test.pkl  train.pkl  valid.pkl
```
### Answer: 4

## Question 3:
Train with autolog mlflow feature and determine number of parameters generated

```bash
tracking_uri = mlflow.set_tracking_uri('sqlite:///hw2.db')
client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
experiment = client.list_experiments()
runs = client.search_runs(experiment[1].experiment_id)
no_of_parameters = len(runs[0].data.params)
```
### Answer: no_of_parameters: 17

## Question 4:
Configure local tracking server
### Answer: default-artifact-root

## Question 5:
Tune hypermeters and find the best validation RMSE

```bash
tracking_uri = mlflow.set_tracking_uri('sqlite:///hw2.db')
client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
experiment = client.list_experiments()
runs = client.search_runs(experiment[1].experiment_id,run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,max_results=1, order_by = ['metrics.rmse'])
round(runs[0].data.metrics['rmse'], 3)
```
### Answer: 6.628

## Question 6:
Model registry and RMSE of test dataset

```bash
# select the model with the lowest test RMSE
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
best_run = client.search_runs(experiment_ids=experiment.experiment_id, run_view_type=ViewType.ACTIVE_ONLY, max_results=1,order_by=['metrics.test_rmse ASC'])[0]
print(f"Run id: {best_run.info.run_id}, test_rmse: {best_run.data.metrics['test_rmse']:.4f}")
# register the best model
model_uri = f'runs:/{best_run.info.run_id}/model'
print("Registering model...")
mlflow.register_model(model_uri=model_uri, name = 'nyc-green-taxi-hw2')
```

### Answer: 6.5479 ~= 6.55
