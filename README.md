# DSRNN_CrowdNav
This repository contains the codes for our paper titled "Decentralized Structural-RNN for Robot Crowd Navigation with Deep Reinforcement Learning" in ICRA 2021. 
For more details, please refer to the 
[arXiv preprint](https://arxiv.org/abs/2011.04820).
For experiment demonstrations, please refer to the [youtube video](https://youtu.be/bYO-1IAjzgY).


## Abstract
Safe and efficient navigation through human crowds is an essential capability for mobile robots. 
Previous work on robot crowd navigation assumes that the dynamics of all agents are known and well-defined. In addition, the performance of previous methods deteriorates in partially observable environments and environments with dense crowds. 
To tackle these problems, we propose decentralized structural-Recurrent Neural Network (DS-RNN), a novel network that reasons about spatial and temporal relationships for robot decision making in crowd navigation. 
We train our network with model-free deep reinforcement learning without any expert supervision. 
We demonstrate that our model outperforms previous methods in challenging crowd navigation scenarios. 
We successfully transfer the policy learned in the simulator to a real-world TurtleBot 2i.

<!--
## Method Overview
<img src="https://i.imgur.com/YOPHXD1.png" width="1000" />
-->

## Setup
1. Install the required python package
```
pip install -r requirements.txt
```

2. Install [OpenAI Baselines](https://github.com/openai/baselines#installation) 
```
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```

3. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
4. Install crowd_sim and crowd_nav into pip
```
pip install -e .
```

## Getting started
This repository is organized in two parts: crowd_sim/ folder contains the simulation environment and
crowd_nav/ folder contains codes for training and testing the policies. The folder pytorchBaselines contains the code
for the network and ppo algorithm.
Details of the simulation framework can be found
[here](crowd_sim/README.md). Below are the instructions for training and testing policies.

### Change configurations
1. Environment configurations: modify crowd_nav/configs/env.config
3. PPO configurations: modify arguments.py 

### Run the code
1. Train a policy. 
```
python train.py 
```

2. Test policies.
```
python test.py 
```
3. Plot training curve.
```
python plot.py
```
(For now, the code only works on GPU)

## Learning Curve
Learning curves of DS-RNN in 360 degrees FoV environment with 5 humans.

<img src="/figures/reward.png" width="600" />
<img src="/figures/loss.png" width="600" />

## Citation
If you find the code or the paper useful for your research, please cite our paper:
```
@misc{liu2020decentralized,
  title={Decentralized Structural-RNN for Robot Crowd Navigation with Deep Reinforcement Learning},
  author={Liu, Shuijing and Chang, Peixin and Liang, Weihang and Chakraborty, Neeloy and Driggs-Campbell, Katherine},
  eprint={arXiv preprint arXiv:2011.04820},
  year={2020}
}
```

## References
[1] C. Chen, Y. Liu, S. Kreiss, and A. Alahi, “Crowd-robot interaction: Crowd-aware robot navigation with attention-based deep reinforcement learning,” in International Conference on Robotics and Automation (ICRA), 2019, pp. 6015–6022.
(Github: https://github.com/vita-epfl/CrowdNav)

[2] I. Kostrikov, “Pytorch implementations of reinforcement learning algorithms,” https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail, 2018.

[3] A. Vemula, K. Muelling, and J. Oh, “Social attention: Modeling attention in human crowds,” in IEEE international Conference on Robotics and Automation (ICRA), 2018, pp. 1–7.
(Github: https://github.com/jeanoh/big)