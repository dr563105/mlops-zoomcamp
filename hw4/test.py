import requests

ride = {
    "PULocationID": 80,
    "DOLocationID": 66,
    "trip_distance": 8900
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())
