import pickle
import os

DATA_PATH = '../data'

def log_dist_and_num_steps(init_dist_to_target, steps_to_completion):
    """
    This function logs the initial distance between agent and target source and number of steps 
    taken to reach target source. The lists are stored in pickle files. The pairs are in parallel
    lists, indexed by the episode number.

    Args:
        init_dist_to_target (List[float]): initial distance between agent and target src (size is number of episodes)
        steps_to_completion (List[int]): number of steps it took for agent to get to source
    """
    dist_url = 'init_dist_to_target.p'
    steps_url = 'steps_to_completion.p'

    # create data folder
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    # write objects
    pickle.dump(init_dist_to_target, open(os.path.join(DATA_PATH, dist_url), 'wb'))
    pickle.dump(steps_to_completion, open(os.path.join(DATA_PATH, steps_url), 'wb'))

    # test
    with open(os.path.join(DATA_PATH, dist_url), 'rb') as f:
        dist = pickle.load(f)
        print('len dist:', len(dist))
        print(dist[:10])

    with open(os.path.join(DATA_PATH, steps_url), 'rb') as f:
        steps = pickle.load(f)
        print('len steps:', len(steps))
        print(steps[:10])

