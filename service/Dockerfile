FROM python:3.11-slim
LABEL author=${AUTHOR}
COPY . ./churn_app
WORKDIR churn_app 
RUN pip3 install -r requirements.txt
EXPOSE ${APP_PORT}
VOLUME /models
CMD uvicorn ml_service.churn_app_prometheus:app --host  0.0.0.0 --port ${APP_PORT}