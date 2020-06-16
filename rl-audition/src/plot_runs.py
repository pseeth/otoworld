import matplotlib.pyplot as plt
import requests
import pandas as pd
import seaborn as sns
import re


port_number = 6006
path = '../models/evaluation_data/'
sns.set(style="whitegrid")
dpi = 300


def create_csv(file_name):
    # Form two urls - for mean reward and cumulative reward

    csv_url_cumul = "http://localhost:{}/data/plugin/scalars/scalars?tag=Reward%2Fcumulative&run={}&format=csv".format(
        port_number, file_name
    )
    csv_url_mean = "http://localhost:{}/data/plugin/scalars/scalars?tag=Reward%2Fmean_per_episode&run={}&format=csv".format(
        port_number, file_name
    )

    # Create CSV for mean rewards
    req = requests.get(csv_url_mean)
    url_content = req.content
    csv_file = open('{}.csv.'.format(path+file_name+'_mean'), 'wb')
    csv_file.write(url_content)
    csv_file.close()

    # Create csv for cumulative rewards
    req = requests.get(csv_url_cumul)
    url_content = req.content
    csv_file = open('{}.csv.'.format(path + file_name+'_cumul'), 'wb')
    csv_file.write(url_content)
    csv_file.close()


def create_plots_single(file_name):

    mean_rewards = pd.read_csv(path+file_name+'_mean.csv')
    cumul_rewards = pd.read_csv(path+file_name+'_cumul.csv')

    sns_plot = sns.relplot(x="Step", y="Value", data=cumul_rewards, kind="line")
    sns_plot.set(xlabel='Step', ylabel='Cumulative Reward')
    sns_plot.savefig(path+file_name+'_cumul.png', dpi=dpi)

    sns_plot = sns.relplot(x="Step", y="Value", data=mean_rewards, kind="line")
    sns_plot.set(xlabel='Step', ylabel='Mean Reward')
    sns_plot.savefig(path+file_name+'_mean.png', dpi=dpi)


def create_plots_multiple(file_names):

    combined_data_mean = pd.DataFrame(columns=['Wall time', 'Step', 'Value', 'Number'])
    combined_data_cumul = pd.DataFrame(columns=['Wall time', 'Step', 'Value', 'Number'])
    for i, file_name in enumerate(file_names):
        mean_rewards = pd.read_csv(path+file_name+'_mean.csv')
        cumul_rewards = pd.read_csv(path+file_name+'_cumul.csv')
        mean_rewards['Number'] = i
        cumul_rewards['Number'] = i
        combined_data_mean = pd.concat([combined_data_mean, mean_rewards])
        combined_data_cumul = pd.concat([combined_data_cumul, cumul_rewards])

    sns_plot = sns.relplot(x="Step", y="Value", data=combined_data_mean, kind="line", hue="Number")
    sns_plot.set(xlabel='Step', ylabel='Mean Reward')


    sns_plot.savefig(path + '_mean_combined.png', dpi=dpi)

    sns_plot = sns.relplot(x="Step", y="Value", data=combined_data_cumul, kind="line", hue="Number")
    sns_plot.set(xlabel='Step', ylabel='Cumulative Reward')
    sns_plot.savefig(path+'_cumul-combined.png', dpi=dpi)


def generate_data_from_log(file_name):
    steps_second = []
    finished_steps = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
        counter = 0
        while counter < len(lines):
            cur_line = lines[counter]
            if 'Episode: ' in cur_line:
                ep_num = int(re.findall(r'\d+', cur_line)[0])
                # Skip validation episodes
                if ep_num % 5 == 0:
                    pass
                else:
                    # Grab the data
                    counter += 2
                    cur_line = lines[counter]
                    finished_count = int(re.findall(r'\d+', cur_line)[0])
                    finished_steps.append(finished_count)
                    counter += 2
                    cur_line = lines[counter]
                    sps = float(re.findall(r'\d+\.\d+', cur_line)[0])
                    steps_second.append(sps)
            counter += 1

    print(len(finished_steps), len(steps_second))
    print(finished_steps)
    print(steps_second)

    return finished_steps, steps_second


if __name__ == '__main__':
    file_names = ["test-exp-5-50eps_test_simp_env_validation-2_15_06_2020-02_33_50",
                  "exp5-200eps-final-run_15_06_2020-06_30_00"]

    # for file_name in file_names:
    #     create_csv(file_name=file_name)
    #     create_plots_single(file_name)
    #
    # create_plots_multiple(file_names)

    generate_data_from_log(file_name=path+'run.txt')