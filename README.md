# give-life-to-agents
Give-life-to-agents is a project for bio-inspired neural network training.
<p float="center">
  <img src="https://user-images.githubusercontent.com/16518993/123286330-ca1a1280-d548-11eb-8789-1b27edaee9a8.gif" width="400" />
  <img src="https://user-images.githubusercontent.com/16518993/123286575-fcc40b00-d548-11eb-9e73-1ec3b465d5ce.gif" width="400" /> 
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
- [x] ANN
- [ ] SNN

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
python run_es.py --config conf/lunarlander.yaml 

# training BiPedalWalker-v3
python run_es.py --config conf/bipedal.yaml 
```

## Test saved model

```bash
# training LunarLander-v2
python test.py --config conf/lunarlander.yaml --ckpt-path <saved-model-dir>
```

NOTE : \<saved-model-dir>  is the path to the directory containing the .pt file, not .pt file.


