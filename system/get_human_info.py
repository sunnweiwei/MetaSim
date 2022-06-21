import json
import os
from parse import parse_belief_state_all
from fill import get_item
from tqdm import tqdm
from collections import defaultdict

def is_success(bs_gen, bs_gt):
    gt_items = get_item(goal)
    model_items = get_item(bs_gen)
    success = True
    for k in gt_items.keys():
        if k != 'hospital':
            if k not in model_items.keys():
                success = False
                break
            else:
                if model_items[k] == []:
                    success = False
                    break
                elif model_items[k][0] not in gt_items[k]:
                    success = False
                    break
    return success

save_path = 'multiwoz_save/'
data_list = defaultdict(list)
total = defaultdict(int)
num_suc = defaultdict(int)
combine_suc = defaultdict(int)
for file in tqdm(os.listdir(save_path)):
    system = file.split('mwoz-')[-1].split('_')[0]
    data = json.load(open(save_path+file, 'r'))
    try:
        sat = data['satisfaction']
        bs_gen = data['state']
        goal = parse_belief_state_all(data['goal'])
        data['eval_success'] = is_success(bs_gen, goal)
        if data['eval_success']:
            num_suc[system] += 1
        data['combine_success'] = (int(sat) / 5 + int(data['eval_success'])) / 2
        combine_suc[system] += data['combine_success']
        total[system] += 1
        data_list[system].append(data)
    except:
        continue
for system in total.keys():
    print(f"system: {system}")
    print(f"eval_success: {num_suc[system] / total[system]}")
    print(f"combine_success: {combine_suc[system] / total[system]}")
    json.dump(data_list[system], open(f'human_eval/{system}.json', 'w'))