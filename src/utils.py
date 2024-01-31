from dataclasses import dataclass
from os import getcwd

import numpy as np


def file_namer(n, r, d, iters, extra=None):
    path = getcwd().split("src")[0]

    return (
        f"{path}/results/N-{n if not isinstance(n, list) else '(' + str(min(n)) + '-' + str(max(n)) + ')x' + str(len(n))}"
        f"__R-{str(r).replace('.', '-') if not isinstance(r, list) else ('(' + str(min(r)) + '-' + str(max(r)) + ')x' + str(len(r))).replace('.', '-')}"
        f"__D-{d if not isinstance(d, list) else '(' + str(min(d)) + '-' + str(max(d)) + ')x' + str(len(d))}"
        f"__I-{iters}{'_' + extra if extra else ''}.pkl"
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
