import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df_for_hire = pd.read_parquet("../input/fhv_tripdata_2021-01.parquet")


df_for_hire['duration_in_sec'] = (df_for_hire['dropOff_datetime'] - df_for_hire['pickup_datetime']).dt.total_seconds()/60.0

categorical = ['PUlocationID', 'DOlocationID']
df_for_hire[categorical].fillna(-1.0,inplace=True)

((df_for_hire['PUlocationID']== -1.0).sum()/df_for_hire.shape[0])*100


df_for_hire = df_for_hire.loc[(df_for_hire['duration_in_sec'] >= 1) & (df_for_hire['duration_in_sec'] <= 60)]

df_for_hire[categorical] = df_for_hire[categorical].astype(str)
dv = DictVectorizer()
features_dict = df_for_hire[categorical].to_dict(orient='records')

X_train = dv.fit_transform(features_dict)



target = 'duration_in_sec'
y_train = df_for_hire[target].to_numpy()
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)

mean_squared_error(y_train, y_pred, squared=False)



df_feb2021 = pd.read_parquet('../input/fhv_tripdata_2021-02.parquet')
df_feb2021['duration_in_sec'] = (df_feb2021['dropOff_datetime'] - df_feb2021['pickup_datetime']).dt.total_seconds()/60.0

df_feb2021 = df_feb2021.loc[(df_feb2021['duration_in_sec'] >= 1) & (df_feb2021['duration_in_sec'] <= 60)]


df_feb2021['PUlocationID'].fillna(-1.0,inplace=True)
df_feb2021['DOlocationID'].fillna(-1.0,inplace=True)
df_feb2021[categorical] = df_feb2021[categorical].astype(str)
test_dict = df_feb2021[categorical].to_dict(orient='records')

X_valid = dv.transform(test_dict)
y_valid = df_feb2021[target].to_numpy()

y_pred1 = lr.predict(X_valid)

mean_squared_error(y_valid, y_pred1, squared=False)


