import numpy as np

min_ep = 0.1
max_ep = 0.8
decay_rate = 0.0005
#
# for i in range(10000):
#     cur_value = min_ep + (max_ep - min_ep)*np.exp(-decay_rate*i)
#     print("Step number {}, Ep Value {}".format(i, cur_value))


degrees = np.deg2rad(30)
cur_degree = degrees
for i in range(20):
    print("Radians: {}, Degrees {}".format(cur_degree, np.rad2deg(cur_degree)))
    cur_degree = round((cur_degree + degrees)%(2*np.pi), 4)


agent_info = [[1.2, 3.5], 0.32423]
print(agent_info)
a, b = agent_info
print(a, b)