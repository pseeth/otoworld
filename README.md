<h1 align="center">OtoWorld</h1>

OtoWorld is an interactive environment in which agents must learn to listen in order to solve navigational tasks. The purpose of OtoWorld is to facilitate reinforcement learning research in computer audition, where agents must learn to listen to the world around them to navigate. 

**Note:** Currently the focus is on audio source separation.

OtoWorld is built on three open source libraries: OpenAI [`gym`](https://gym.openai.com/) for environment and agent interaction, [`pyroomacoustics`](https://github.com/LCAV/pyroomacoustics) for ray-tracing and acoustics simulation, and [`nussl`](https://github.com/nussl/nussl) for training deep computer audition models. OtoWorld is the audio analogue of GridWorld, a simple navigation game. OtoWorld can be easily extended to more complex environments and games. 

To solve one episode of OtoWorld, an agent must move towards each sounding source in the auditory scene and "turn it off". The agent receives no other input than the current sound of the room. The sources are placed randomly within the room and can vary in number. The agent receives a reward for turning off a source. 

[Read the OtoWorld Paper here](https://arxiv.org/abs/2007.06123)
<br>


![OtoWorld Environment](otoworld.png)


## Installation 
Clone the repository
```
git clone https://github.com/pseeth/otoworld.git
```
Create a conda environment: 
```
conda create -n otoworld python==3.7
``` 
Activate the environment:
```
conda activate otoworld
```
Install requirements:
```
pip install -r requirements.txt
```
Install ffmpeg from conda distribution (Note: Pypi distribution of ffmpeg is outdated):
```
conda install ffmpeg
```
If using a **CUDA-enabled GPU (highly recommended)**, install Pytorch `1.4` from official source:  
```
pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
```
otherwise: 
```
pip install torch==1.4.0 torchvision==0.5.0
```

## Additional Installation Notes - Linux
* Linux users may need to install the sound file library if it is not present in the system. It can be done using the following command: 
```
sudo apt-get install libsndfile1
```

This should take care of a common `musdb` error.

## Demo and Tutorial 
You can get familiar with OtoWorld using our tutorial notebook: [Tutorial Notebook](https://github.com/pseeth/otoworld/blob/master/notebooks/tutorial.ipynb).

Run 
```
jupyter notebook
``` 
and navigate to `notebooks/tutorial.ipynb`.

## Experiments
You can view (and run) examples of experiments:
```
cd experiments/

python experiment1.py
```

Please create your own experiments and see if you can win OtoWorld! You will need a GPU running CUDA to be able to perform any meaningful experiments. 

## Is It Running Properly?
You should a message indicating the experiment is running, such as this:
```
------------------------------ 
- Starting to Fit Agent
------------------------------- 
```

## Citing
```
@inproceedings {otoworld
    author = {Omkar Ranadive and Grant Gasser and David Terpay and Prem Seetharaman},
    title = "OtoWorld: Towards Learning to Separate by Learning to Move",
    journal = "Self Supervision in Audio and Speech Workshop, 37th International Conference on Machine Learning ({ICML} 2020), Vienna, Austria",
    year = 2020
}
```
