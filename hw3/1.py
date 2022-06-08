import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def read_data(path):
    return pd.read_parquet(path)

def pre_process(data_df, categorical):
    data_df['duration_in_sec'] = (data_df['dropOff_datetime'] - data_df['pickup_datetime']).dt.total_seconds()/60.0
    data_df = data_df[(data_df['duration_in_sec'] >= 1) & (data_df['duration_in_sec'] <= 60)].copy()
    data_df[categorical] = data_df[categorical].fillna(-1.0).astype(int).astype(str)
    return data_df 

def train_model(df_train_pp, categorical, target):
    dv = DictVectorizer()
    train_dict = df_train_pp[categorical].to_dict(orient='records')

    X_train = dv.fit_transform(train_dict)
    y_train = df_train_pp[target].to_numpy()
   
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    y_pred = lr_model.predict(X_train)
    training_mse = mean_squared_error(y_train, y_pred,squared=False)
    print(f'Training RMSE is {training_mse}')
    
    return lr_model, dv

def validate_model(df_valid_pp, model, dict_vectorizer, categorical, target):
    val_dict = df_valid_pp[categorical].to_dict(orient='records') 
    X_val = dict_vectorizer.transform(val_dict)
    Y_val = df_valid_pp[target].to_numpy()

    y_pred = model.predict(X_val)
    return mean_squared_error(Y_val, y_pred, squared=False)

def main(train_path:str=r'../input/fhv_tripdata_2021-01.parquet',val_path:str=r'../input/fhv_tripdata_2021-02.parquet'):
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_pp = pre_process(df_train, categorical)

    df_valid = read_data(val_path)
    df_valid_pp = pre_process(df_valid, categorical)
    
    target = 'duration_in_sec'

    model, dict_vectorizer = train_model(df_train_pp, categorical, target)
    rmse = validate_model(df_valid_pp, model, dict_vectorizer, categorical, target)
    print(f'Validation RMSE is {rmse}')

if __name__ == '__main__':
    main()