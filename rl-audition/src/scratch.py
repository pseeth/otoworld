import numpy as np

min_ep = 0.1
max_ep = 0.8
decay_rate = 0.0005
#
# for i in range(10000):
#     cur_value = min_ep + (max_ep - min_ep)*np.exp(-decay_rate*i)
#     print("Step number {}, Ep Value {}".format(i, cur_value))


a = np.random.choice(100, 5)
a = np.append(a, 99)
print(a)