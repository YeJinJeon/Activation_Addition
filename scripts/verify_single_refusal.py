from jaxtyping import Float, Int
import tqdm
import pandas as pd

from tmp import *

"""
run with hook
args: steering_vector for each layer

1. Activation Addition
input: new positive prompts not used in generating steering vector
    1) generate hook_fn with pre-defined steering vector
    2) run with hook in specific layer
"""

def ablation_hook_fn_from_steer_vec(direction, coeff=5):
    direction = direction.unsqueeze(0).cuda()
    def prompt_hook(resid_pre, hook):
        nonlocal direction
        resid_pre = resid_pre.squeeze(0)
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(resid_pre) 
        resid_pre -= (resid_pre @ direction.T) * direction 
        resid_pre += coeff * direction
        return resid_pre.unsqueeze(0)
    return prompt_hook


def addition_hook_fn_from_steer_vec(steer_vec):
    steer_vec = steer_vec.unsqueeze(0).unsqueeze(0).cuda()
    def prompt_hook(resid_pre, hook):
        assert (resid_pre.shape[2] == steer_vec.shape[2]
                ),"The embeddind size of token and the steering vector doesn't match"
        resid_pre[0, -1:, :] =  (
            steer_vec + resid_pre[0, -1:, :]
        )
        return resid_pre
    return prompt_hook


def run_steering(prompt, steer_layer: str, steer_vec, steer_type):
    # generate activation hook 
    hook_fns: Dict[str, List[Callable]] = {}
    if steer_type == "add":
        hook_fns[steer_layer] = [addition_hook_fn_from_steer_vec(steer_vec)]
    elif steer_type == "ablate":
        # TODO: modify to steer all layers and tokens
        hook_fns[steer_layer] = [ablation_hook_fn_from_steer_vec(steer_vec)]

    fwd_hooks = [
            (name, hook_fn)
            for name, hook_fns in hook_fns.items()
            for hook_fn in hook_fns
    ]

    # steer response
    sampling_kwargs: Dict[str, Union[float, int]] = {
        "temperature": 1.0,
        "top_p": 0.3,
        "freq_penalty": 1.0,
    }
    with model.hooks(fwd_hooks=fwd_hooks):  # type: ignore
        tokenized_prompts = model.to_tokens(prompt)
        completions = model.generate( 
            input=tokenized_prompts, 
            max_new_tokens=50,
            verbose=False,
            **sampling_kwargs,
        )
    response = model.to_string(completions)
    return response

    
if __name__ == "__main__":
    pos_data_path = "/home/yejeon/Activation_Addition/data/alpaca_data.json"
    neg_data_path = "/home/yejeon/Activation_Addition/data/advbench_harmful_behaviors.csv"
    steering_vector_path = "/home/yejeon/Activation_Addition/data/steer_vectors.pt"
    data_num = 30
    data_offset = 200

    # read steering_vector
    steer_vector_dict = torch.load(steering_vector_path)
    
    ## Ablation: Neg -> Pos
    # red neg data
    data = pd.read_csv(neg_data_path)
    neg_eval_data= data['goal'].tolist()[data_offset:data_offset+data_num]

    # evaluation on prompts
    result_dict: Dict[str, List[str]] = {}
    for act_name, act_vector in steer_vector_dict.items():
        result_dict[act_name] = []
        for prompt_idx in tqdm.tqdm(range(len(neg_eval_data))):
            output = run_steering(neg_eval_data[prompt_idx], act_name, act_vector, steer_type="ablate")
            result_dict[act_name].append(output)
        # Serializing json
        result_json = json.dumps(result_dict, indent=4)
        with open("/home/yejeon/Activation_Addition/results/ablation_steer_result.json", "w") as outfile:
            outfile.write(result_json)

    # ## Activation_Addition : Pos -> Neg
    # # read pos data
    # f = open(pos_data_path)
    # data = json.load(f)
    # pos_eval_data = []
    # for i in range(data_num):
    #     pos_eval_data.append(data[data_offset+i]["instruction"]+data[data_offset+i]["input"])
    
    # # evaluation on prompts
    # result_dict: Dict[str, List[str]] = {}
    # for act_name, act_vector in steer_vector_dict.items():
    #     result_dict[act_name] = []
    #     for prompt_idx in tqdm.tqdm(range(len(pos_eval_data))):
    #         output = run_steering(pos_eval_data[prompt_idx], act_name, act_vector, steer_type="add")
    #         result_dict[act_name].append(output)
    #     # Serializing json
    #     result_json = json.dumps(result_dict, indent=4)
    #     with open("/home/yejeon/Activation_Addition/results/activation_addition_steer_result.json", "w") as outfile:
    #         outfile.write(result_json)
            