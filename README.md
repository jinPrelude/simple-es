# simple-es
### Simple implementations of multi-agent evolutionary strategies using minimal dependencies.
<p float="center">
  <img src="https://user-images.githubusercontent.com/16518993/126927433-583913b2-b329-47c4-a002-4b3f9d1e0923.gif" width="250" /> 
  <img src="https://user-images.githubusercontent.com/16518993/126900857-12ff3f52-0a3a-4670-aea3-aa7b73c2a04b.gif" width="250"  />
  <img src="https://user-images.githubusercontent.com/16518993/126922639-5baa4176-f85d-4642-a6b3-ddd94ed56448.gif" width="250" />
</p>

**Simple-es is designed to help you quickly understand evolutionary learning through code, so we considered easy-to-understand code structure first, yet has strong features.**

*Latest Check Date: Aug.11.2021*

\
This project has 4 main features:
1. evolutionary strategies with gym environment
2. recurrent neural newtork support
3. Pettingzoo multi-agent environment support
4. wandb sweep parameter search support

**NOTE: If you want a NEAT algorithm that has the same design pattern with [simple-es](https://github.com/jinPrelude/simple-es) and performs more powerful distributed processing using mpi4py, visit [pyNeat](https://github.com/jinPrelude/pyNeat).**

## Algorithms
We Implemented three algorithms below:
- **simple_evolution**: Use Gaussian noise for offspring generation and apply the average weight of the offssprings to the weight of the next parent(mu) model.
- **simple_genetic**: Use Gaussian noise to generate offspring for N parent models, and adopt the N models with the highest performance among offsprings as the next parent model. No mutation process implemented.
- **[openai_es](https://openai.com/blog/evolution-strategies/)**: Evolutionary strategy proposed by openAI in 2017 to solve problems of reinforcement learning. Visit the link for more information.

## Recurrent Neural Network with POMDP environments.
Recurrent ANN(GRU) is also implemented by default. The use of the gru module can be set in the config file. For environment, LunarLander and CartPole support POMDP setting.
```python
network:
  gru: True
env:
  name: "CartPole-v1"
  pomdp: True
```
config file ```conf/lunarlander_openai.yaml``` is applied to run in a POMDP setting, and it learns very well. You can try by running the command below:
```bash
python run_es.py --cfg-path conf/lunarlander_openai.yaml
```
### POMDP CartPole benchmarks
GRU agent with simple-evolution strategy(green) got perfect score (500) in POMDP CartPole environment, whereas ANN agent(yellow) scores nearly 60, failed to learn POMDP CartPole environment. GRU agent with simple-genetic strategy(purple) also shows poor performance.

<img src=https://user-images.githubusercontent.com/16518993/125189883-4d3fa600-e275-11eb-9311-1a3cce3d5041.png width=600>

## Pettingzoo Multi-Agent Environment
Three [pettingzoo](https://github.com/PettingZoo-Team/PettingZoo) envionments are currently implemented: simple_spread, waterworld, multiwalker. But you can easily add other pettingzoo enviornments by ```modifying envs/pettingzoo_wrapper.py```. You can try simple_spread environment by running the command below:
```bash
python run_es.py --cfg-path conf/simplespread.yaml
```

## Wandb Sweep hyperparameter search
Wandb Sweep is a hyperparameter search tool serviced by [wandb](https://wandb.ai/home). **It automatically finds the best hyperparameters for selected environment and strategy.** hyperparameter for LunarLander with POMDP setting(```conf/lunarlander_openai.yaml```) is a good example of finding the hyperparameters quickly through the sweep function.

<img src="https://user-images.githubusercontent.com/16518993/126923452-6f1fce73-1c8b-466d-90ac-c474c9e04cb7.png" width="600">

There is an example config file in the path ```sweep_config/``` you can try wandb sweep.
It can be run as follows:
```bash
> wandb sweep sweep_config/lunarlander_openaies.yaml
# command above will automatically create a sweep project and then print the execution command.
# ex) Wandb: Run sweep agent with: wandb agent <sweep prject name>
> wandb agent <sweep prject name>
```
Visit [here](https://docs.wandb.ai/guides/sweeps) for more information about wandb sweep.
## Installation
### prerequisite
You need following library:
```
> sudo apt install swig # for box2d-py
```

We recommand you to install in virtual environment to avoid any dependency issues.
```bash
# recommend python==3.8.10
> git clone https://github.com/jinPrelude/simple-es.git
> cd simple-es
> pip install -r requirements.txt
```

### Increase thread limit
To train offsprings > 100, You may increase the system's python thread limit. Since it's python's fundamental issue, you can increase by modifying `/etc/security/limits.conf`
```bash
> sudo vim /etc/security/limits.conf
```
and add the codes below:
```
*               soft    nofile          65535
*               hard    nofile          65535
```
save and quit the file by vim command: `esc` + `:` + `wq` + `enter`, and reboot your computer.
## Train

```bash
# training LunarLander-v2
> python run_es.py --cfg-path conf/lunarlander.yaml 

# training BiPedalWalker-v3
> python run_es.py --cfg-path conf/bipedal.yaml --log
```

You need [wandb](https://wandb.ai/) account for logging. Wandb provides various useful logging features for free.

## Test saved model

```bash
# training LunarLander-v2
> python test.py --cfg-path conf/lunarlander.yaml --ckpt-path <saved-model-dir> --save-gif
```


