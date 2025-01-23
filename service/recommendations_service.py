# FastAPI-микросервис для выдачи рекомендаций, который:

#     принимает запрос с данными клиента и выдаёт рекомендации
#     смешивает онлайн- и офлайн-рекомендации

import numpy as np
from utils import (
    rec_store,
    preprocess_data,
    dedup_ids,
    describe_by_name,
    name_by_id,
    filter_ids,
    cat_cols,
    num_cols,
    mlc_model,
    kmeans_model,
    one_hot_drop,
    standart_scaler,
    kmeans_parquet,
    ncodpers_dict,
)
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from dotenv import load_dotenv

load_dotenv()

# Получаем уже созданный логгер "uvicorn.error", чтобы через него можно было логировать собственные сообщения в тот же поток,
# в который логирует и uvicorn
logger = logging.getLogger("uvicorn.error")
logging.basicConfig(filename="rec_history.log", level=logging.INFO)
logging.info("Started")


# Функция ниже, которая передаётся как параметр FastAPI-объекту, выполняет свой код только при запуске приложения и при его остановке.
# При запуске приложения загружаем персональные и ТОП-рекомендации
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting")

    # для оффайн-рекомендаций: автозагрузка перс-рекомендаций
    rec_store.load(
        type="personal",
        columns=["ncodpers", "names", "score"],
    )

    # для оффайн-рекомендаций: автозагрузка классификаторских-рекомендаций
    rec_store.load(
        type="classify",
        columns=["ncodpers", "names", "score"],
    )

    # для оффайн-рекомендаций: автозагрузка кластерных рекомендаций
    rec_store.load(
        type="default",
        columns=["names", "score"],
    )

    yield
    logging.info("Stopping")


# создаём приложение FastAPI
app = FastAPI(title="FastAPI-микросервис для выдачи рекомендаций", lifespan=lifespan)


@app.post("/recommendations", name="Получение рекомендаций для клиента")
async def recommendations(ncodpers_dict: dict = ncodpers_dict, k: int = 5):
    """
    Возвращает список рекомендаций длиной k для клиента ncodpers.

    На вход принимает json с данными клиента:

        {'fecha_dato': '2015-01-28',
        'ncodpers': 1375586,
        'ind_empleado': 'N',
        'pais_residencia': 'ES',
        'sexo': 'H',
        'age': 35,
        'fecha_alta': '2015-01-12',
        'ind_nuevo': 0,
        ...
        'ind_valo_fin_ult1': 0,
        'ind_viv_fin_ult1': 0,
        'ind_nomina_ult1': 0,
        'ind_nom_pens_ult1': 0,
        'ind_recibo_ult1': 0}
    """

    logging.info("-------------------------------------")

    # Список оффлайн рекомендаций
    recs_offline = rec_store.get(ncodpers_dict["ncodpers"], k)

    # Список онлайн рекомендаций
    recs_online_mlc = await get_online_mlc(ncodpers_dict)
    recs_online_mlc = recs_online_mlc["recs"]

    recs_online_kmeans = await get_online_kmeans(ncodpers_dict)
    recs_online_kmeans = recs_online_kmeans["recs"]

    # Минимальная длина списков
    min_length = min(
        len(recs_offline),
        len(recs_online_mlc),
        len(recs_online_kmeans),
    )

    logging.info(f"recs_offline {describe_by_name(recs_offline)}")
    logging.info(f"recs_online_mlc {describe_by_name(recs_online_mlc)}")
    logging.info(f"recs_online_kmeans {describe_by_name(recs_online_kmeans)}")

    # Чередуем элементы из списков, пока позволяет минимальная длина
    recs_blended = []
    for i in range(min_length):
        recs_blended.append([recs_online_mlc[i]])
        recs_blended.append([recs_online_kmeans[i]])
        recs_blended.append([recs_offline[i]])

    # Добавляем оставшиеся элементы в конец
    recs_blended.append(recs_offline[min_length:])
    recs_blended.append(recs_online_mlc[min_length:])
    recs_blended.append(recs_online_kmeans[min_length:])

    # Удаление дубликатов
    recs_blended = dedup_ids(sum(recs_blended, []))

    # Удаление активных продуктов
    recs_blended = filter_ids(ncodpers_dict["ncodpers"], recs_blended)

    # Оставляем только первые k-рекомендаций
    recs_blended = recs_blended[:k]

    # Список продуктов итогового рекоммендатора
    if recs_blended:
        logging.info(f"Список итоговых рекомендаций: {describe_by_name(recs_blended)}")

    return {"recs": recs_blended}


@app.post("/get_online_mlc")
async def get_online_mlc(ncodpers_dict: dict = ncodpers_dict) -> dict:
    """
    Возвращает список онлайн-рекомендаций MLC по словарю с данными клиента

    Формат входных данных:

        {'fecha_dato': '2015-01-28',
        'ncodpers': 1375586,
        'ind_empleado': 'N',
        'pais_residencia': 'ES',
        'sexo': 'H',
        'age': 35,
        'fecha_alta': '2015-01-12',
        'ind_nuevo': 0,
        ...
        'ind_valo_fin_ult1': 0,
        'ind_viv_fin_ult1': 0,
        'ind_nomina_ult1': 0,
        'ind_nom_pens_ult1': 0,
        'ind_recibo_ult1': 0}
    """

    logging.info("-------------------------------------")

    recs_id = []

    # Общее преобразование входных данных, согласно EDA
    ncodpers_dict = preprocess_data(ncodpers_dict)

    if ncodpers_dict:

        # Преобразование признаков согласно пайплайну MLC
        feature_list = cat_cols + num_cols
        feature_list = list(dict((w, ncodpers_dict[w]) for w in feature_list).values())

        # Инференс
        recs = mlc_model.predict_proba(feature_list)

        # Сортировка по убыванию
        recs_scores = list(np.sort(recs)[::-1])
        recs_id = list(np.argsort(recs)[::-1])

        recs_id = name_by_id(recs_id)

        logging.info(f"Список онлайн-рекомендаций MLC: ")
        for x, y in zip(describe_by_name(recs_id), recs_scores):
            logging.info(f"{x}, {y}")
    else:
        logging.info(f"Список онлайн-рекомендаций MLC: []")
        recs_id = []

    return {"recs": recs_id}


@app.post("/get_online_kmeans")
async def get_online_kmeans(ncodpers_dict: dict = ncodpers_dict) -> dict:
    """
    Возвращает список онлайн-рекомендаций K-MEANS по словарю с данными клиента

    Формат входных данных:

        {'fecha_dato': '2015-01-28',
        'ncodpers': 1375586,
        'ind_empleado': 'N',
        'pais_residencia': 'ES',
        'sexo': 'H',
        'age': 35,
        'fecha_alta': '2015-01-12',
        'ind_nuevo': 0,
        ...
        'ind_valo_fin_ult1': 0,
        'ind_viv_fin_ult1': 0,
        'ind_nomina_ult1': 0,
        'ind_nom_pens_ult1': 0,
        'ind_recibo_ult1': 0}
    """

    logging.info("-------------------------------------")

    recs_id = []

    # Общее преобразование входных данных, согласно EDA
    ncodpers_dict = preprocess_data(ncodpers_dict)

    if ncodpers_dict:

        # Кодирование признаков согласно пайплайну кластеризации
        feature_cats_list = list(dict((w, ncodpers_dict[w]) for w in cat_cols).values())
        feature_nums_list = list(dict((w, ncodpers_dict[w]) for w in num_cols).values())
        drop_res = one_hot_drop.transform([feature_cats_list])
        scaler_res = standart_scaler.transform([feature_nums_list])
        features_coded = list(scaler_res[0]) + list(drop_res[0])

        # Инференс
        cluster_id = kmeans_model.predict([features_coded])
        cluster_id = list(cluster_id)[0]

        recs_id = kmeans_parquet[kmeans_parquet["labels"] == cluster_id][
            "names"
        ].to_list()

        recs_scores = kmeans_parquet[kmeans_parquet["labels"] == cluster_id][
            "score"
        ].to_list()

        logging.info(f"Список онлайн-рекомендаций K-MEANS: ")
        for x, y in zip(describe_by_name(recs_id), recs_scores):
            logging.info(f"{x}, {y}")

    else:
        logging.info(f"Список онлайн-рекомендаций K-MEANS: []")
        recs_id = []

    return {"recs": recs_id}


@app.get("/load_recommendations", name="Загрузка рекомендаций из файла")
async def load_recommendations(rec_type: str = "classify"):
    """
    Загружает оффлайн-рекомендации из файла (на случай, если файлы рекомендаций обновились)
    """

    logging.info("-------------------------------------")

    if rec_type == "personal":
        columns = ["ncodpers", "names", "score"]
    elif rec_type == "classify":
        columns = ["ncodpers", "names", "score"]
    else:
        columns = ["names", "score"]
    rec_store.load(
        type=rec_type,
        columns=columns,
    )


@app.get("/get_statistics", name="Получение статистики по рекомендациям")
async def get_statistics():
    """
    Выводит статистику по имеющимся счётчикам
    """

    logging.info("-------------------------------------")

    return rec_store.stats()


# запуск сервиса
# uvicorn recommendations_service:app
# INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
