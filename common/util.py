import random
import numpy as np


def get_random_sample_ids(length, K):
    import random
    ids_list = []
    for i in range(length):
        ids_list.append(i)
    ids = random.sample(ids_list, K)

    if K == 1:
        ids = ids[0]
    return ids



if __name__ == '__main__':
    print(get_random_sample_ids(10, 1))
