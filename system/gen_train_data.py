import os
import pickle
from parse import parse_belief_state
import json
from tqdm import tqdm

save_path = "output/interaction_train/"

def write_pkl(obj, filename):
    dirname = '/'.join(filename.split('/')[:-1])
    os.makedirs(dirname, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def read_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def in_goal(gt_bs:dict, gen_bs:dict):
    for domain in gen_bs.keys():
        if domain not in gt_bs.keys():
            return False
        for slot in gen_bs[domain].keys():
            if slot not in gt_bs[domain].keys() or gen_bs[domain][slot] != gt_bs[domain][slot]:
                return False
    return True

def parse_goal(goal:dict):
    flat_goal = []
    for key in goal.keys():
        att = []
        for k, v in goal[key].items():
            att.append(f'{k} = {v}')
        if att:
            bs = f'{key} ' + ' ; '.join(att)
            flat_goal.append(bs)
    return ' | '.join(flat_goal).lower() if flat_goal else 'none'

train_data = []
train_data_clean = []
total_dig = 0
for file in tqdm(os.listdir(save_path)):
    data = read_pkl(f"{save_path}{file}")
    if not data["success"]:
        continue
    total_dig += 1
    goal = parse_belief_state(data["goal"])
    for idx in range(len(data["context"])):
        if idx % 2 == 1 and '[END]' not in data['context'][idx]:
            history = data["context"][:idx]
            reply = data["origin"][idx // 2].replace('system:', 'system :')
            cur_state = data["belief"][idx // 2]
            belief = "belief : " + parse_goal(cur_state)
            cur_data = {
                "history": history,
                "reply": reply,
                "belief": belief
            }
            train_data.append(cur_data)
            if in_goal(goal, cur_state):
                train_data_clean.append(cur_data)
print(f"total_dig: {total_dig}")
print(f"total_data: {len(train_data)}")
print(f"total_data_clean: {len(train_data_clean)}")
json.dump(train_data, open("data/domain_0.1_total_train.json", 'w'))
json.dump(train_data_clean, open("data/domain_0.1_total_clean_train.json", 'w'))

