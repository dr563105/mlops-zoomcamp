FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim 

# python:3.9.7-slim

WORKDIR /app
# COPY [ "model2.bin", "model.bin" ]
RUN pip install -U pip
RUN pip install pipenv
COPY ["Pipfile","Pipfile.lock", "./"]
RUN pipenv install --system --deploy
COPY ["starter.py", "./"]
ENTRYPOINT ["python3", "starter.py"]
