import requests
import time
import random

for i in range(100):

    user_id = random.randint(1, 1000)
    has_elevator = random.randint(0, 1)
    floors_total = random.randint(1, 50)

    print(f"user_id: {user_id}")
    print(f"has_elevator: {has_elevator}")
    print(f"floors_total: {floors_total}")

    params = {
        "building_id": 6220,
        "build_year": 1965,
        "building_type_int": 6,
        "latitude": 55.717113,
        "longitude": 37.78112,
        "ceiling_height": 2.64,
        "flats_count": 84,
        "floors_total": floors_total,
        "has_elevator": has_elevator,
        "floor": 9,
        "kitchen_area": 9.9,
        "living_area": 19.9,
        "rooms": 1,
        "is_apartment": 0,
        "total_area": 35.099998,
    }

    response = requests.post(
        f"http://localhost:1702/api/churn/?user_id={user_id}", json=params
    )

    print(f"response: {response}")

    if i == 10:
        time.sleep(3)

    time.sleep(1)
