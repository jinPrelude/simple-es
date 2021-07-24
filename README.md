# simple-es
### Simple implementations of multi-agent evolutionary strategies using minimal dependencies.
<p float="center">
  <img src="https://user-images.githubusercontent.com/16518993/123286330-ca1a1280-d548-11eb-8789-1b27edaee9a8.gif" width="300" />
  <img src="https://user-images.githubusercontent.com/16518993/123286575-fcc40b00-d548-11eb-9e73-1ec3b465d5ce.gif" width="300" /> 
</p>
Simple-es is designed to help you quickly understand evolutionary learning through code, so we considered easy-to-understand code structure first, yet has strong features.

\
This project has 4 main features:
1. evolutionary strategies with gym environment
2. recurrent neural newtork support
3. multi-agent environment support(pettingzoo)
4. wandb sweep parameter search support

**NOTE: This repo is archived for stable reproducibility. Visit [torch-es](https://github.com/jinPrelude/mpi-es.git) for more advanced features!! It uses mpi4py to speed up training, and planning to implement more bio-inspired algorithms like NEAT families, and hebbian learning!**

## Algorithms
We Implemented three algorithms below:
- **simple_evolution**: Use Gaussian noise for offspring generation and apply the average weight of the offssprings to the weight of the next parent(mu) model.
- **vanilla genetic srtategy**: Use Gaussian noise to generate offspring for N parent models, and adopt the N models with the highest performance among offsprings as the next parent model. No mutation process implemented.
- **[OpenAI ES](https://openai.com/blog/evolution-strategies/)**: Evolutionary strategy proposed by openAI in 2017 to solve problems of reinforcement learning. Visit the link for more information.

## Recurrent Neural Network with POMDP environments.
Recurrent ANN(GRU) is also implemented by default. The use of the gru module can be set in the config file. For environment, LunarLander and CartPole support POMDP setting.
```python
network:
  gru: True
env:
  name: "CartPole-v1"
  pomdp: True
```
config file ```conf/lunarlander_openai.yaml``` is applied to run in a POMDP setting. You can run the config file by running the command below:
```bash
python run_es.py --cfg-path conf/lunarlander_openai.yaml
```
### POMDP CartPole benchmarks
GRU agent with simple-evolution strategy(green) got perfect score (500) in POMDP CartPole environment, whereas ANN agent(yellow) scores nearly 60, failed to learn POMDP CartPole environment. GRU agent with simple-genetic strategy(purple) also shows poor performance.
<img src=https://user-images.githubusercontent.com/16518993/125189883-4d3fa600-e275-11eb-9311-1a3cce3d5041.png width=600>

## Multi-Agent Environment
Three envionments are currently implemented: simple_spread, waterworld, multiwalker. But you can easily add other pettingzoo enviornments by ```modifying envs/pettingzoo_wrapper.py```. You can try simple_spread environment by running the command below:
```bash
python run_es.py --cfg-path conf/simplespread.yaml
```

## Wandb Sweep hyperparameter search
...

## Installation

```bash
# recommend python==3.8.10
git clone https://github.com/jinPrelude/simple-es.git
cd simple-es
pip install -r requirements.txt
```

## Train

```bash
# training LunarLander-v2
python run_es.py --cfg-path conf/lunarlander.yaml 

# training BiPedalWalker-v3
python run_es.py --cfg-path conf/bipedal.yaml --log
```

You need [wandb](https://wandb.ai/) account for logging. Wandb provides various useful logging features for free.

## Test saved model

```bash
# training LunarLander-v2
python test.py --cfg-path conf/lunarlander.yaml --ckpt-path <saved-model-dir> --save-gif
```


