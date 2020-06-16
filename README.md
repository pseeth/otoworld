<h1 align="center">OtoWorld</h1>

Otoworld is an interactive environment for training agents for the task of audio separation. <br>

**Paper Submission:** [TBD]

## Installation 

* Initialize a conda environment with **Python 3.7** as follows: 
```
conda create -n rl-audio python=3.7 pip
```
* Install the packages in requirements.txt as follows: 
```
pip install -r requirements.txt
```
* Install ffmpeg from conda distribution [Note: Pypi distribution of ffmpeg is outdated]
```
conda install ffmpeg
```
* Build nussl from the git repository as follows: 
```
pip install -U git+git://github.com/nussl/nussl
```
* Install Pytorch **1.3** or **1.4** from official source 

## Additional installation notes 
* Linux users may need to install the sound file library if it is not present in the system. It can be done using the following command: 
```
sudo apt-get install libsndfile1
```

## Tutorial 
You can get familiar with this OtoWorld using our tutorial notebook: [Tutorial Notebook](https://github.com/pseeth/rl_for_audition/blob/master/rl-audition/notebooks/tutorial.ipynb)



