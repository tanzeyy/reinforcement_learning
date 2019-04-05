import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from env import *
from utils import *

plt.rcParams['figure.dpi'] = 200
k = 10
agents_num = 500

def train_agents(agents, envs, **kwargs):
    testbed = list(zip(agents, envs))
    data = train(testbed, **kwargs)
    return data

env = NonstationaryEnv
data = dict()
kws = {
    "baseline, alpha: 0.1": {
        'init_q_value': 0,
        'alpha': 0.1,
        'steps': 1000
    },
    "baseline, alpha: 0.4": {
        'init_q_value': 0,
        'alpha': 0.4,
        'steps': 1000
    }
}
kws_no_baseline = {
    "no baseline, alpha: 0.1": {
        'init_q_value': 0,
        'alpha': 0.1,
        'steps': 1000
    },
    "no baseline, alpha: 0.4": {
        'init_q_value': 0,
        'alpha': 0.4,
        'steps': 1000
    }
}

envs = [env(k, mean=4, var=1) for _ in range(agents_num)]

agent = GradientAscent
for _k, _v in kws.items():
    agents = [agent(k, 
                    init_q_value=_v.get('init_q_value'), 
                    alpha=_v.get('alpha')) for _ in range(agents_num)]
    _v.pop('init_q_value')
    _v.pop('alpha')
    data[_k] = train_agents(agents, envs, **_v)

# No baseline situation
agent = GradientAscentNoBaseline
for _k, _v in kws_no_baseline.items():
    agents = [agent(k, 
                    init_q_value=_v.get('init_q_value'), 
                    alpha=_v.get('alpha')) for _ in range(agents_num)]
    _v.pop('init_q_value')
    _v.pop('alpha')
    data[_k] = train_agents(agents, envs, **_v)

fig, ax = plt.subplots(figsize=(10, 5))
opt_prop_curve = cal_opt_prop(data)
plot_curve(data=opt_prop_curve, 
           ax=ax, 
           title='', 
           x_label='Steps', 
           y_label='Opt. Act. Prop.')
plt.show()
 