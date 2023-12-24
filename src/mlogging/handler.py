import json
import subprocess
import argparse
import requests


def parse_args():
    """
    Parse the arguments imputed when running the script from the command line
    If you wish to run this in the editor vary the default values as desired.
    Returns:
        The arguments accessible via dot notation
    """
    n = 100
    parser = argparse.ArgumentParser(description="Graph simulation inputs")
    parser.add_argument(
        "graphs",
        type=int,
        nargs="?",
        const=n,
        default=n,
        help="The number of graphs to be simulated",
    )
    args = parser.parse_args()
    return args


def start_logger():
    args = parse_args()
    subprocess.run(["python", "src/mlogging/mlogger.py", f"{args.graphs}"])


def update_status(box_index, new_color):
    api_url = f"http://127.0.0.1:5000/api/box/{box_index}"
    data = {"color": new_color}
    headers = {"Content-Type": "application/json"}

    response = requests.put(api_url, data=json.dumps(data), headers=headers)

    if response.status_code != 200:
        print(f"Failed to change color. Status code: {response.status_code}")
