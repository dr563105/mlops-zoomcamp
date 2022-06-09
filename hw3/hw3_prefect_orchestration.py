import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from datetime import date, datetime

import pickle

import prefect
from prefect import flow, task
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

@task(name="getting file's path")
def get_paths(fetch_date):
    """
    Takes date(if given) as input and returns the paths of training and validation files
    """
    date_format = "%Y-%m-%d"

    if fetch_date == None:
        fetch_date = date.today()
    elif type(fetch_date) == str:
        fetch_date = datetime.strptime(fetch_date, date_format)

    train_file_date = f"{fetch_date.year}-{fetch_date.month-2:02d}"
    valid_file_date = f"{fetch_date.year}-{fetch_date.month-1:02d}"

    download_file_path = r"../input/fhv_tripdata_"
    train_complete_path = f"{download_file_path}{train_file_date}.parquet"
    valid_complete_path = f"{download_file_path}{valid_file_date}.parquet"

    logger = prefect.get_run_logger()
    logger.info(f"Training file path: {train_complete_path}")
    logger.info(f"Validation file path: {valid_complete_path}")

    return train_complete_path, \
            valid_complete_path, \
            datetime.strftime(fetch_date, date_format)


@task(name='Reading data from parquet files')
def read_data(path):
    """
    Takes file path as input and returns pandas dataframe
    """
    logger = prefect.get_run_logger()
    logger.info('reading data using logger...')
    return pd.read_parquet(path)


@task(name='Preprocessing')
def pre_process(data_df, categorical):
    """
    Preprocesses the data
    """
    logger = prefect.get_run_logger()
    logger.info('preprocessing...')
    data_df['duration_in_sec'] = (data_df['dropOff_datetime'] - data_df['pickup_datetime']).dt.total_seconds()/60.0
    data_df = data_df[(data_df['duration_in_sec'] >= 1) & (data_df['duration_in_sec'] <= 60)].copy()
    data_df[categorical] = data_df[categorical].fillna(-1.0).astype(int).astype(str)
    return data_df


@task(name='Training model')
def train_model(df_train_pp, categorical, target):
    """
    Takes the preprocessed data and trains the model. Returns both the model and dict vectorizer
    """
    logger = prefect.get_run_logger()
    logger.info('training model...')
    dv = DictVectorizer()
    train_dict = df_train_pp[categorical].to_dict(orient='records')

    X_train = dv.fit_transform(train_dict)
    y_train = df_train_pp[target].to_numpy()

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_pred = lr_model.predict(X_train)
    training_mse = mean_squared_error(y_train, y_pred,squared=False)
    logger.info(f'Training RMSE is {training_mse}')

    return lr_model, dv


@task(name='validating model')
def validate_model(df_valid_pp, model, dict_vectorizer, categorical, target):
    """
    Uses the model, dict vectorizer from the training module and validates
    the model with the validation dataframe
    """
    logger = prefect.get_run_logger()
    logger.info('validating...')
    val_dict = df_valid_pp[categorical].to_dict(orient='records')
    X_val = dict_vectorizer.transform(val_dict)
    Y_val = df_valid_pp[target].to_numpy()

    y_pred = model.predict(X_val)
    rmse = mean_squared_error(Y_val, y_pred, squared=False)
    logger.info(f'Validation RMSE is {rmse}')


@flow(name='nyc taxi analysis_hw3')
def main(fetch_date:str=f"2021-08-15"):
    """
    Main function
    """

    logger = prefect.get_run_logger()
    categorical = ['PUlocationID', 'DOlocationID']
    train_path, val_path, fetch_date = get_paths(fetch_date=fetch_date).result()


    df_train = read_data(train_path)
    df_train_pp = pre_process(df_train, categorical)

    df_valid = read_data(val_path)
    df_valid_pp = pre_process(df_valid, categorical)

    target = 'duration_in_sec'

    model, dict_vectorizer = train_model(df_train_pp, categorical, target).result()
    validate_model(df_valid_pp, model, dict_vectorizer, categorical, target)

    model_filename = f"model-{fetch_date}.bin"
    dv_filename = f"dv-{fetch_date}.b"

    with open(model_filename, "wb") as f_out:
        logger.info("Pickling model...")
        pickle.dump(model,f_out)

    with open(dv_filename, "wb") as f_out:
        logger.info("Pickling dict vectorizer...")
        pickle.dump(dict_vectorizer,f_out)

# DeploymentSpec(
#     flow=main,
#     name="model_training",
#     # schedule=CronSchedule(cron="0 9 15 * *"), #runs at 9 am UTC on 15th of each month
#     flow_runner=SubprocessFlowRunner(),
#     tags=["hw3-orchestration"]
# )


if __name__ == '__main__':
    main() #While running deployment, comment above and current lines and uncomment DeploymentSpec block
