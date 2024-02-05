import os
from dataclasses import dataclass
from os import getcwd
import json

import numpy as np


def file_clean_up(temp_file, new_file):
    for file in os.listdir(temp_file):
        with open(f"{temp_file}{file}", "a") as f:
            f.write(']')
        with open(f"{temp_file}{file}", "r+") as f:
            data = json.load(f)
            for datum in data:
                append_json_lines(new_file, datum)

    with open(new_file, "a") as f:
        f.write('\n' + ']')

    delete_folder(temp_file)


def delete_folder(folder_path):
    """Deletes a folder and all its contents recursively.

    Args:
        folder_path: The path to the folder to be deleted.
    """

    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # Check if the folder is empty
    if not os.listdir(folder_path):
        print(f"Folder '{folder_path}' is already empty.")
        return

    # Delete the folder contents recursively
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isdir(file_path):
            delete_folder(file_path)
        else:
            os.remove(file_path)

    # Delete the folder itself
    os.rmdir(folder_path)
    print(f"Folder '{folder_path}' deleted successfully.")


def numpy_to_list(arr):
    """Recursively converts nested NumPy arrays into Python lists.

    Args:
        arr: The NumPy array or nested structure to convert.

    Returns:
        A Python list containing the converted elements.
    """

    if isinstance(arr, np.ndarray):
        return arr.tolist()  # Convert NumPy array to list
    elif isinstance(arr, (list, tuple)):
        return [numpy_to_list(item) for item in arr]  # Recursively convert elements
    else:
        return arr  # Non-array types are left unchanged


def append_json_lines(filename, new_data):
    """Appends a JSON object to a JSON Lines file.

    Args:
        filename (str): The name of the JSON Lines file.
        new_data (dict): The new JSON object to append.
    """
    exists = os.path.exists(filename)
    with open(filename, 'a' if os.path.exists(filename) else 'w') as file:
        json_string = json.dumps(new_data)  # Convert data to JSON string
        if not exists:  # Check if file exists
            file.write("[" + '\n')  # Write opening square bracket if new file
            file.write(json_string)  # Append to file with newline
        else:
            file.write("," + "\n" + json_string)


def file_namer(n, r, d, iters, extra=None, json=False):
    path = getcwd().split("src")[0]

    return (
        f"{path}/results/N-{n if not isinstance(n, list) else '(' + str(min(n)) + '-' + str(max(n)) + ')x' + str(len(n))}"
        f"__R-{str(r).replace('.', '-') if not isinstance(r, list) else ('(' + str(min(r)) + '-' + str(max(r)) + ')x' + str(len(r))).replace('.', '-')}"
        f"__D-{d if not isinstance(d, list) else '(' + str(min(d)) + '-' + str(max(d)) + ')x' + str(len(d))}"
        f"__I-{iters}{'_' + extra if extra else ''}.{'json' if json else 'pkl'}"
    )


def nrange(start, stop, num):
    return list(np.linspace(start, stop, num, dtype=int))


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


@dataclass
class Node:
    indx: int
    position: np.ndarray

    def to_dict(self):
        return {"node": self.indx, "position": self.position}


@dataclass
class Order:
    node: int
    height: int
    depth: int

    @property
    def order(self):
        return self.height + self.depth

    def to_dict(self):
        return {"node": self.node, "height": self.height, "depth": self.depth}


@dataclass
class Relatives:
    node: int
    children: list
    parents: list

    def to_dict(self):
        return {"node": self.node, "children": self.children, "parents": self.parents}


@dataclass
class Paths:
    longest: list
    greedy_e: list
    greedy_m: list
    random: list
    shortest: list

    def to_dict(self):
        return {
            "longest": self.longest,
            "greedy_e": self.greedy_e,
            "greedy_m": self.greedy_m,
            "random": self.random,
            "shortest": self.shortest,
        }
