FROM python:3.11-slim
LABEL author=${AUTHOR}
COPY . ./app
WORKDIR app 
RUN pip3 install -r requirements.txt
EXPOSE ${APP_PORT}
VOLUME /models
CMD uvicorn recommendations_service:app --host  0.0.0.0 --port ${APP_PORT}

# cd ./service/
# docker image build . --tag final_project:0
# docker container run --publish 1702:1702 --volume=./models:/app/models --env-file .env final_project:0