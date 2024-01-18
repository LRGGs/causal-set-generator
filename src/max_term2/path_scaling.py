import numpy as np
import pandas
import os

df = pandas.read_pickle("../../results/N-(1000-10000)x201__R-0-10__D-2__I-1")

n =
longest_paths = []
for i in range(200):
    df = df.loc[int(n) : int(n + 44)]
    print(df)
    #filtered_slice = slice.where("slice.in_longest == 1")
    #print(filtered_slice)
    n += 45