import time

import numpy as np

alpha = 0.049
beta = 0.0052

max_index = 300
min_index = -250
base = 1.01
n_iter = 10000


def cal_next_recall_halflife(halflife, recall):
    if recall == 1:
        return halflife / (1 - alpha)
    else:
        return halflife / (1 + beta)


def cal_halflife_index(s):
    return max(min(max_index - min_index - 1, round(np.log(s) / np.log(base)) - min_index), 0)


def cal_optimal_policy():
    halflife_list = np.array([np.power(base, i) for i in range(min_index, max_index)])
    index_len = len(halflife_list)
    cost_list = np.array([1000.0 if i != index_len - 1 else 0.0 for i in range(index_len)])
    used_interval_list = np.array([0.0 for _ in range(index_len)])
    recall_list = np.array([0.0 for _ in range(index_len)])
    h0 = 0.1
    h0_index = cal_halflife_index(h0)
    for i in range(n_iter):
        h0_stress = cost_list[h0_index]
        for h_index in reversed(range(0, index_len - 1)):
            halflife = halflife_list[h_index]
            interval_list = list(halflife * np.log((i - 0.01) / 100) / np.log(0.5) for i in range(50, 101))
            for ivl in interval_list:
                recall = np.exp2(- ivl / halflife)
                h_recall = cal_next_recall_halflife(halflife, 1)
                h_recall_index = cal_halflife_index(h_recall)
                h_forget = cal_next_recall_halflife(halflife, 0)
                h_forget_index = cal_halflife_index(h_forget)
                exp_stress = recall * cost_list[h_recall_index] + (1 - recall) * cost_list[h_forget_index] + 1
                if exp_stress < cost_list[h_index]:
                    cost_list[h_index] = exp_stress
                    used_interval_list[h_index] = ivl
                    recall_list[h_index] = recall

        diff = h0_stress - cost_list[h0_index]
        if diff < 0.1:
            print("finished at iter:", i)
            break

    return used_interval_list


start_time = time.time()
used_interval_list = cal_optimal_policy()
print(f"time cost: {time.time() - start_time: .2f}s")
