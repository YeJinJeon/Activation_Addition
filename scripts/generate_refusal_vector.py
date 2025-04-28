import json
import pandas as pd
import io
from tmp import *

pos_data_path = "/home/yejeon/Activation_Addition/data/alpaca_data.json"
neg_data_path = "/home/yejeon/Activation_Addition/data/advbench_harmful_behaviors.csv"
data_num = 100

# read pos data
f = open(pos_data_path)
data = json.load(f)
pos_data = []
for i in range(data_num):
     pos_data.append(data[i]["instruction"]+data[i]["input"])

# read neg_data
data = pd.read_csv(neg_data_path)
neg_data= data['goal'].tolist()[:data_num]

# collect cache of last token on each prompt for every layer
max_response_length = 1
cache_target = "pre_post_act"
layer_num = 16
# get cache for positive data
pos_cache = [[] for _ in range(layer_num)]
layer_names = [] # extract layer names for later
for prompt in pos_data:
    _, cache = generate_with_cache_n_hooks(prompt, None, max_response_length, target_act=cache_target)
    layer_names = list(cache[0].keys())
    for i, act_name in enumerate(cache[0].keys()):
         activation = torch.unsqueeze(cache[0][act_name][0][-1],0)
         pos_cache[i].append(activation)
# get cache for positive data
neg_cache = [[] for _ in range(layer_num)]
for prompt in neg_data:
    _, cache = generate_with_cache_n_hooks(prompt, None, max_response_length, target_act=cache_target)
    for i, act_name in enumerate(cache[0].keys()):
         activation = torch.unsqueeze(cache[0][act_name][0][-1],0) # last token
         neg_cache[i].append(activation)

# calculate steering vector per each layer
steer_vectors = {}
for i in range(layer_num):
     pos_mean = torch.mean(torch.cat(pos_cache[i]), 0)
     neg_mean = torch.mean(torch.cat(neg_cache[i]), 0)
     diff_mean = neg_mean - pos_mean
     steer_vectors[layer_names[i]] = diff_mean
print(steer_vectors)
for name in steer_vectors:
     print(steer_vectors[name].shape)

# save steering vector
torch.save(steer_vectors, "/home/yejeon/Activation_Addition/data/steer_vectors.pt")


         
         

