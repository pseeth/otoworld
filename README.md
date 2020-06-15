# <center>OtoWorld<center> 
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

## Tutorial 
You can get familiar with this environment using our tutorial notebook: [Tutorial Notebook](https://github.com/pseeth/rl_for_audition/blob/master/rl-audition/notebooks/tutorial.ipynb)



