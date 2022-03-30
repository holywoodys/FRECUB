from matplotlib import pyplot as plt
import pandas as pd
import json
import numpy as np
import random
import time
from FreqCB import FreqCB


def main(beta, num_stages, num_users, d, m, L, pj, users, items, user_items, item_features, iterations):
    seed = int(time.time() * 100) % 399
    print("Seed = %d" % seed)
    np.random.seed(seed)
    random.seed(seed)

    ps = [list(np.ones(num_users) / num_users)]

    p = ps[0]
    model = FreqCB(beta, nu = num_users, d = d, p = p, num_stages = num_stages, users = users, items = items, user_items = user_items, item_features = item_features)
    start = time.time()
    cumulative_reward = model.run()
    end = time.time()
    execution_time = round(end - start, 1)
    execution_time = (str(round(execution_time / 60, 1)) + "m" if execution_time > 60 else str(execution_time) + "s")

    print("FreqCB: {}  {}".format(cumulative_reward[iterations], execution_time))
    return cumulative_reward


def Read_data(data):

    user_features = None
    name1 = "directory/Synthetic datasets/"
    item_features = np.load(name1 + data + "/items_features.npy")
    users = np.load(name1 + data + "/users.npy")
    users = users.tolist()
    items = np.load(name1 + data + "/items.npy")
    items = items.tolist()

    with open(name1 + data + "/user_items_json", "r") as f:
        stre = f.read()
        user_items = json.loads(stre)

    return users, items, user_items, item_features, user_features


data = "Vatest2-1000i-100u-20"

users, items, user_items, item_features, user_features = Read_data(data)
n_arms = len(items)
n_users = len(users)
d = len(item_features[0])
print(d)

number = 5

para = [0.1, 0.3, 0.5, 0.7, 1]
# para = [0.001,0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1]

index_columns = ["CTR", "STD", "MAX", "MIN"]
index_rows = [data for p in range(len(para))]
arr = np.zeros((len(para), len(index_columns)))
iterations = 50000
stages = int(np.log2(50000) + 1)
for i in range(len(para)):
    result0 = []
    for j in range(number):
        cc_re = main(beta=para[i], num_stages=stages, num_users=n_users, d=d, m=5, L=20, pj=0, users=users, items=items, user_items=user_items,
             item_features=item_features, iterations=iterations)
        result0.append(cc_re[iterations])

    result = [_ / iterations for _ in result0]
    
    arr[i][0] = np.mean(result)
    arr[i][1] = np.std(result)
    arr[i][2] = np.max(result)
    arr[i][3] = np.min(result)
    print(para[i], result, arr[i])
    df = pd.DataFrame(arr, index=index_rows, columns=index_columns)
    # df.to_csv("directory/" + data + "_" + "FreqCB-syn.csv")

print(df)


