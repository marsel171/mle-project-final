import requests
import time
import random
import json

with open("test_user.json") as f:
    json_data = json.load(f)

for i in range(100):

    ncodpers = random.randint(100000, 1000000)
    age = random.randint(18, 60)
    antiguedad = random.randint(1, 10)
    cod_prov = random.randint(1, 100)
    renta = random.randint(10000, 300000)

    print(f"ncodpers: {ncodpers}")
    print(f"age: {age}")
    print(f"antiguedad: {antiguedad}")
    print(f"cod_prov: {cod_prov}")
    print(f"renta: {renta}")

    json_data["ncodpers"] = ncodpers
    json_data["age"] = age
    json_data["antiguedad"] = antiguedad
    json_data["cod_prov"] = cod_prov
    json_data["renta"] = renta

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }

    params = {
        "k": "5",
    }

    response = requests.post(
        "http://0.0.0.0:1702/recommendations",
        params=params,
        json=json_data,
    )

    print(f"response: {response}")

    if i == 10:
        time.sleep(3)

    time.sleep(1)

# cd ./service/
# python test_requests.py
