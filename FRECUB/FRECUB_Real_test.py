from matplotlib import pyplot as plt
import pandas as pd

import json
import numpy as np
import random
import time
from FRECUB import FRECUB


def main(beta, num_stages, num_users, d, m, L, pj, users, items, user_items, item_features, iterations):
    seed = int(time.time() * 100) % 399
    print("Seed = %d" % seed)
    np.random.seed(seed)
    random.seed(seed)

    ps = [list(np.ones(num_users) / num_users)]

    p = ps[0]
    model = FRECUB(beta, nu=num_users, d=d, p=p, num_stages=num_stages, users=users, items=items, user_items=user_items,
                   item_features=item_features)
    start = time.time()
    cumulative_reward = model.run()
    end = time.time()
    execution_time = round(end - start, 1)
    execution_time = (str(round(execution_time / 60, 1)) + "m" if execution_time > 60 else str(execution_time) + "s")

    print("SCLUB: {}  {}".format(cumulative_reward[iterations], execution_time))
    return cumulative_reward


def Read_data(data):

    user_features = None
    name1 = "directory"
    item_features = np.load(name1 + data + "/items_features.npy")
    users = np.load(name1 + data + "/users.npy")
    users = users.tolist()
    items = np.load(name1 + data + "/items.npy")
    items = items.tolist()

    with open(name1 + data + "/user_items_json", "r") as f:
        stre = f.read()
        user_items = json.loads(stre)

    return users, items, user_items, item_features, user_features


data = "Last.FM"

users, items, user_items, item_features, user_features = Read_data(data)
n_arms = len(items)
n_users = len(users)
d = len(item_features[0])

number = 5

para = [0.1, 0.3, 0.5, 0.7, 1]
result_SCLUB = []
index_columns = ["CTR", "STD", "MAX", "MIN"]

index_rows = [data for p in range(len(para))]
arr = np.zeros((len(para), len(index_columns)))
iterations = 50000
stages = int(np.log2(iterations) + 1)
for i in range(len(para)):
    result0 = []
    lis = [[] for p in range(number)]
    for j in range(number):
        cc_re = main(beta=para[i], num_stages=stages, num_users=n_users, d=d, m=5, L=20, pj=0, users=users, items=items, user_items=user_items,
             item_features=item_features, iterations=iterations)
        for k in range(len(cc_re)):
            if (k + 1) % 1000 == 0:
                lis[j].append(cc_re[k] / (k + 1))
        # result0.append(cc_re[iterations])
        # plt.plot(cc_re[0:5001], label="SCLUB")
    print(para[i])
    print(lis)



