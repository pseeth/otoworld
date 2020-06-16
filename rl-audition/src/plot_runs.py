import matplotlib.pyplot as plt
import requests
import pandas as pd
import seaborn as sns

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



if __name__ == '__main__':
    file_names = ["test-exp-5-50eps_test_simp_env_validation-2_15_06_2020-02_33_50",
                  "exp5-200eps-final-run_15_06_2020-06_30_00"]

    for file_name in file_names:
        create_csv(file_name=file_name)
        create_plots_single(file_name)

    create_plots_multiple(file_names)