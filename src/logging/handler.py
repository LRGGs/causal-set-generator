import requests
import json


def update_status(box_index, new_color):
    api_url = f'http://127.0.0.1:5000/api/box/{box_index}'
    data = {'color': new_color}
    headers = {'Content-Type': 'application/json'}

    response = requests.put(api_url, data=json.dumps(data), headers=headers)

    if response.status_code != 200:
        print(f"Failed to change color. Status code: {response.status_code}")
        print(response.json())