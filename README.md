# give-life-to-agents
Give-life-to-agents is a project for bio-inspired neural network training.
<p float="center">
  <img src="https://user-images.githubusercontent.com/16518993/123286330-ca1a1280-d548-11eb-8789-1b27edaee9a8.gif" width="300" />
  <img src="https://user-images.githubusercontent.com/16518993/123286575-fcc40b00-d548-11eb-9e73-1ec3b465d5ce.gif" width="300" /> 
</p>

## Algorithms
### learning strategies
- [x] vanilla evolution srtategy
- [x] vanilla genetic srtategy
- [ ] CMA-ES
- [ ] MAPPO(Multi Agent RL)
- [ ] [OpenAI ES](https://openai.com/blog/evolution-strategies/)
- [ ] [WANN](https://arxiv.org/abs/1906.04358)
- [ ] [hebbian plasticity](https://arxiv.org/abs/2007.02686)

### networks
- [x] ANN(+ GRU)
- [ ] Indirect Encoding
- [ ] SNN

## Recurrent ANN with POMDP CartPole
Recurrent ANN(GRU) is also implemented by default. The use of the gru module can be set in the config file. For environment, LunarLander and CartPole support POMDP setting.
```python
network:
  gru: True
env:
  name: "LunarLanderContinuous-v2"
  pomdp: True
```
### POMDP CartPole benchmarks
GRU agent with simple-evolution strategy(<span style="color:green">green</span>) scores over 200 in POMDP CartPole environment, whereas ANN agent(<span style="color:yellow">yellow</span>) scores nearly 60, failed to successfully learn POMDP CartPole environment. GRU agent with simple-genetic strategy(<span style="color:purple">purple</span>) also shows poor performance.
<img src=https://user-images.githubusercontent.com/16518993/124372010-42f43980-dcc2-11eb-848f-eecaa9c7f30c.png width=600>


## Installation

```bash
# recommend python==3.8.10
git clone https://github.com/jinPrelude/give-life-to-agents.git
cd give-life-to-agents
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


