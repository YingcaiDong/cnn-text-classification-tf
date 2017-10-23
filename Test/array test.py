import numpy as np

zh_data = ['你','好','中','国','喜','迎','十','八','大']
data1 = [1, 10, 100, 1000, 10000, 100000, 2, 20, 200]
data = np.array(list(zip(data1, zh_data)))

shuffle_index = np.random.permutation(np.arange(len(data)))
shuffled_data = data[shuffle_index]
print(shuffled_data)