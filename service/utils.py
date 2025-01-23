import pandas as pd
from catboost import CatBoostClassifier
import logging
import pickle
import boto3
from dotenv import load_dotenv
import os
import io
import json

load_dotenv()

# Получаем уже созданный логгер "uvicorn.error", чтобы через него можно было логировать собственные сообщения в тот же поток,
# в который логирует и uvicorn
logger = logging.getLogger("uvicorn.error")


# Подключаемся к хранилищу данных для загрузки рекомендаций и моделей
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
S3_SERVICE_NAME = "s3"
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
session = boto3.session.Session()
s3 = session.client(
    service_name=S3_SERVICE_NAME,
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# Модель MLC (для выдачи онлайн-рекомендаций)
cat_cols = [
    "sexo",
    "ind_nuevo",
    "indext",
    "canal_entrada",
    "cod_prov",
    "ind_actividad_cliente",
    "segmento",
]
num_cols = ["age", "antiguedad", "renta"]
mlc_model = CatBoostClassifier(
    cat_features=cat_cols,
    loss_function="MultiLogloss",
    iterations=10,
    learning_rate=1,
    depth=2,
)
obj_mlc_model = s3.get_object(Bucket=BUCKET_NAME, Key=os.environ.get("KEY_MLC_MODEL"))
mlc_model = mlc_model.load_model(stream=io.BytesIO(obj_mlc_model["Body"].read()))

# Модель K-means (для выдачи онлайн-рекомендаций) и его энкодеры
obj_kmeans_model = s3.get_object(
    Bucket=BUCKET_NAME, Key=os.environ.get("KEY_KMEANS_MODEL")
)
kmeans_model = pickle.load(io.BytesIO(obj_kmeans_model["Body"].read()))

obj_one_hot_drop = s3.get_object(
    Bucket=BUCKET_NAME, Key=os.environ.get("KEY_KMEANS_MODEL_ONE_HOT_DROP")
)
one_hot_drop = pickle.load(io.BytesIO(obj_one_hot_drop["Body"].read()))

obj_standart_scaler = s3.get_object(
    Bucket=BUCKET_NAME, Key=os.environ.get("KEY_KMEANS_MODEL_STANDART_SCALER")
)
standart_scaler = pickle.load(io.BytesIO(obj_standart_scaler["Body"].read()))

# Таблица с кластерами для выдачи онлайн-рекомендаций
obj_kmeans_parquet = s3.get_object(
    Bucket=BUCKET_NAME, Key=os.environ.get("KEY_KMEANS_PARQUET")
)
kmeans_parquet = pd.read_parquet(io.BytesIO(obj_kmeans_parquet["Body"].read()))

# Справочник products_catalog.parquet с описанием продуктов
obj_products_catalog_parquet = s3.get_object(
    Bucket=BUCKET_NAME, Key=os.environ.get("KEY_PRODUCTS_CATALOG_PARQUET")
)
products_catalog = pd.read_parquet(
    io.BytesIO(obj_products_catalog_parquet["Body"].read())
)

# Таблица с наличием продуктов клиентов в последнем месяце
obj_last_activity_parquet = s3.get_object(
    Bucket=BUCKET_NAME, Key=os.environ.get("KEY_LAST_ACTIVITY_PARQUET")
)
last_activity = pd.read_parquet(io.BytesIO(obj_last_activity_parquet["Body"].read()))


# Загрузка тестового клиента с его данными
obj_test_user_json = s3.get_object(
    Bucket=BUCKET_NAME, Key=os.environ.get("KEY_TEST_USER_JSON")
)
ncodpers_dict = json.load(io.BytesIO(obj_test_user_json["Body"].read()))


# Класс готовых рекомендаций (при запуске загружаются готовые рекомендации, а затем и отдаются при вызове /recommendations)
class Recommendations:
    """
    Методы:

    load - загружает рекомендации указанного типа из файла.
    get - отдаёт рекомендации, а если таковые не найдены, то рекомендации по умолчанию.
    stats - выводит статистику по имеющимся счётчикам.
    """

    def __init__(self):

        self._recs = {"personal": None, "classify": None, "default": None}
        self._stats = {
            "request_personal_count": 0,  # счетчик персональных рекомендаций (ALS)
            "request_classify_count": 0,  # счетчик рекомендаций по классификатору (MLC)
            "request_default_count": 0,  # счетчик топ-рекомендаций (K-means)
        }

    def load(self, type, **kwargs):
        """
        Загружает рекомендации из файла

        type == "personal" - персональные (при помощи ALS)
        type == "classify" - классификаторские (при помощи MLC)
        type == "default" - топ-рекомендации (при помощи K-means)
        """

        logger.info(f"Loading recommendations, type: {type}")

        if type == "personal":
            obj_personal_als_parquet = s3.get_object(
                Bucket=BUCKET_NAME, Key=os.environ.get("KEY_PERSONAL_ALS_PARQUET")
            )
            self._recs[type] = pd.read_parquet(
                io.BytesIO(obj_personal_als_parquet["Body"].read()), **kwargs
            ).set_index("ncodpers")
        elif type == "classify":
            obj_mlc_parquet = s3.get_object(
                Bucket=BUCKET_NAME, Key=os.environ.get("KEY_MLC_PARQUET")
            )
            self._recs[type] = pd.read_parquet(
                io.BytesIO(obj_mlc_parquet["Body"].read()), **kwargs
            ).set_index("ncodpers")
        else:
            obj_kmeans_parquet = s3.get_object(
                Bucket=BUCKET_NAME, Key=os.environ.get("KEY_KMEANS_PARQUET")
            )
            self._recs[type] = pd.read_parquet(
                io.BytesIO(obj_kmeans_parquet["Body"].read()), **kwargs
            )

        logger.info(f"Loaded")

    def get(self, user_id: int, k: int = 5):
        """
        Возвращает список рекомендаций для пользователя
        """

        recs = []

        # Поиск из персональных рекомендаций
        try:
            recs = (
                self._recs["personal"]
                .loc[user_id]
                .sort_values(by="score", ascending=False)
            )
            recs = recs["names"].to_list()[:k]
            self._stats["request_personal_count"] += 1
            logger.info(f"Found {len(recs)} personal recommendations!")
            pers = 1
            mlc = 1
        except:
            pers = 0

        # Поиск из классификаторских рекомендаций
        if not pers:
            try:
                recs = (
                    self._recs["classify"]
                    .loc[user_id]
                    .sort_values(by="score", ascending=False)
                )
                recs = recs["names"].to_list()[:k]
                self._stats["request_classify_count"] += 1
                logger.info(f"Found {len(recs)} classify recommendations!")
                mlc = 1
            except:
                mlc = 0

        # Поиск из кластерных рекомендаций
        if not mlc:
            try:
                recs = self._recs["default"]
                recs = recs.drop_duplicates(subset=["names"], keep="first").sort_values(
                    by="score", ascending=False
                )
                recs = recs["names"].to_list()[:k]
                self._stats["request_default_count"] += 1
                logger.info(f"Found {len(recs)} TOP-recommendations!")
                top = 1
            except:
                top = 0

        if not recs:
            logger.error("No recommendations found")
            recs = []

        return recs

    def stats(self):

        logger.info("Stats for recommendations")
        for name, value in self._stats.items():
            logger.info(f"{name:<30} {value} ")
        print(self._stats)
        return self._stats


# загрузка готовых оффлайн-рекомендаций
rec_store = Recommendations()


def preprocess_data(ncodpers_dict: dict):
    """
    Преобразование входных данных
    """

    if pd.isnull(ncodpers_dict["sexo"]):
        print("No sexo! No recommendations!")
        ncodpers_dict = []
    else:

        ncodpers_dict["ind_nuevo"] = int(ncodpers_dict["ind_nuevo"])
        ncodpers_dict["ind_actividad_cliente"] = int(
            ncodpers_dict["ind_actividad_cliente"]
        )

        if not ncodpers_dict["indext"]:
            ncodpers_dict["indext"] = "N"

        if not ncodpers_dict["canal_entrada"]:
            ncodpers_dict["canal_entrada"] = "NAN"

        try:
            ncodpers_dict["cod_prov"] = int(ncodpers_dict["cod_prov"])
        except:
            ncodpers_dict["cod_prov"] = 0

        if not ncodpers_dict["segmento"]:
            ncodpers_dict["segmento"] = "NAN"

        ncodpers_dict["antiguedad"] = int(ncodpers_dict["antiguedad"])
        if ncodpers_dict["antiguedad"] < 0:
            ncodpers_dict["antiguedad"] = 0

        try:
            ncodpers_dict["renta"] = int(ncodpers_dict["renta"])
            if ncodpers_dict["renta"] > 350000:
                ncodpers_dict["renta"] = 350000
        except:
            ncodpers_dict["renta"] = 66964.6

    return ncodpers_dict


def describe_by_name(names):
    """
    Находим описание продуктов из справочника по их идентификатору
    """

    return [
        products_catalog[products_catalog["names"] == x]["describe"].to_list()[0]
        for x in names
    ]


def name_by_id(id_list):
    """
    Находим описание продуктов из справочника по их индексу
    """

    return [products_catalog.iloc[x]["names"] for x in id_list]


def filter_ids(ncodpers, names):
    """
    Фильтрует список идентификаторов, оставляя только те продукты, которых не было у клиента в предыдущем месяце
    """

    active_names = last_activity[last_activity["ncodpers"] == ncodpers][
        "names"
    ].to_list()

    return [x for x in names if x not in active_names]


def dedup_ids(ids):
    """
    Дедублицирует список идентификаторов, оставляя только первое вхождение
    """

    seen = set()
    ids = [id for id in ids if not (id in seen or seen.add(id))]

    return ids
