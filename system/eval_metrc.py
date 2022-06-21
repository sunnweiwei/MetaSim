import json
import re

from parse import parse_decoding_results_direct
from fill import get_item
from parse import parse_belief_state_all
from evaluation import eval_all
from tqdm import tqdm

def eval_slot_match(predicts, answers, skip_null=True):
    match = []
    for pred, ans in zip(predicts, answers):
        if len(ans) == 0 and skip_null:
            continue
        match.append(int(all([slot in pred for slot in ans])))
    return {'slot_match': sum(match) / len(match) * 100}

# output_file = ['data-100_context-15.json', 'data-100_context-5.json', 'data-100_context-1.json', 'data-10_context-15.json', 'data-1_context-15.json']
gt = json.load(open('output/gt.json', 'r'))
data = json.load(open('data/multi-woz/data.json', 'r'))

def parse_goal(goal):
    flat_goal = []
    for key in goal:
        if key in ['topic', 'message'] or len(goal[key]) == 0:
            continue
        att = []
        for cat in ['info', 'book']:
            if cat not in goal[key]:
                continue
            for k, v in goal[key][cat].items():
                # k = k.replace('_', ' ')
                att.append(f'{k} = {v}')
        bs = f'domain = {key} ; ' + ' ; '.join(att)
        flat_goal.append(bs)
    return ' | '.join(flat_goal).lower()

output_file = ['data-100_context-15', 'data-100_context-3', 'data-100_context-1',
               'data-10_context-15', 'data-1_context-15', 'mwoz-recommend-0.4', 'mwoz-recommend-0.1']
total = 0
num_success = 0
for file_name in output_file:
    print(f"eval {file_name}:")
    if 'context' in file_name:
        max_turn = file_name.split('-')[-1]
    else:
        max_turn = 15
    test_datas = json.load(open(f'data/test.soloist_filled_{max_turn}.json', 'r'))
    predictions = json.load(open('output_new/' + file_name + '.json', 'r'))
    bs = json.load(open('output_new/' + file_name + '_bs.json', 'r'))
    # slot
    # preds = []
    cache = {}
    current_name = 'SNG0073.json'
    slot_ans = []
    pre_res = []
    for idx, pred in tqdm(enumerate(predictions)):
        test_data = test_datas[idx]
        file = test_data['name']
        gt_res = test_data["reply"]
        slot_list = re.findall(r"(\[\w+\])", gt_res)
        slot_ans.append(slot_list)
        res, _ =  parse_decoding_results_direct(pred)
        pre_res.append(res)
        if file != current_name:
            goal = parse_belief_state_all(parse_goal(data[current_name]['goal']))
            # print(current_name, cache)
            # if 'train' in goal.keys():
            #     current_name = file
            #     cache = {}
            #     continue
            # print(goal)
            # print(cache)
            gt_items = get_item(goal)
            model_items = get_item(cache)
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
            # print(success)
            total += 1
            if success:
                num_success += 1
            current_name = file
            cache = {}
        res_bs = bs[idx]
        for domain in res_bs.keys():
            if domain not in cache.keys():
                cache[domain] = res_bs[domain]
            elif len(res_bs[domain].keys()) > len(cache[domain].keys()):
                cache[domain] = res_bs[domain]
    goal = parse_belief_state_all(parse_goal(data[current_name]['goal']))
    # print(current_name, cache)
    gt_items = get_item(goal)
    model_items = get_item(cache)
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
    # print(success)
    total += 1
    if success:
        num_success += 1
    print(f'success: {num_success / total}')
    # print(f"slot: {eval_slot_match(pre_res, slot_ans)}")
    # print(eval_all(pre_res, gt))
