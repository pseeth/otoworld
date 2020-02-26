import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = '../data'
DIST_URL = 'init_dist_to_target.p'
STEPS_URL = 'steps_to_completion.p'

def log_dist_and_num_steps(init_dist_to_target, steps_to_completion):
    """
    This function logs the initial distance between agent and target source and number of steps 
    taken to reach target source. The lists are stored in pickle files. The pairs are in parallel
    lists, indexed by the episode number.

    Args:
        init_dist_to_target (List[float]): initial distance between agent and target src (size is number of episodes)
        steps_to_completion (List[int]): number of steps it took for agent to get to source
    """

    # create data folder
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    # write objects
    pickle.dump(init_dist_to_target, open(os.path.join(DATA_PATH, DIST_URL), 'wb'))
    pickle.dump(steps_to_completion, open(os.path.join(DATA_PATH, STEPS_URL), 'wb'))


def plot_dist_and_steps():
    with open(os.path.join(DATA_PATH, DIST_URL), 'rb') as f:
        dist = pickle.load(f)
        avg_dist = np.mean(dist)
        print(dist[:20])
        #print('Avg initial dist:', )

    with open(os.path.join(DATA_PATH, STEPS_URL), 'rb') as f:
        steps = pickle.load(f)
        avg_steps = np.mean(steps)
        #print('Avg # steps:', avg_steps)

    plt.scatter(dist, np.log(steps))
    plt.title('Number of Steps and Initial Distance')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Log(# of Steps to Reach Target)')
    plt.text(5, 600, "Avg Steps: " + str(int(avg_steps)), size=15, rotation=0.,
         ha="right", va="top",
         bbox=dict(boxstyle="square",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )
    plt.text(5, 500, "Avg Init Dist: " + str(int(avg_dist)), size=15, rotation=0.,
         ha="right", va="top",
         bbox=dict(boxstyle="square",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )
    plt.show()


def main():
    plot_dist_and_steps()

# main()