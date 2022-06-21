import json
import re

from parse import parse_decoding_results_direct
from fill import get_item
from parse import parse_belief_state_all
from tqdm import tqdm

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

gt = json.load(open('output/gt.json', 'r'))
data = json.load(open('data/multi-woz/data.json', 'r'))

output_file = ['data-100_context-15', 'data-100_context-3', 'data-100_context-1',
               'data-10_context-15', 'data-1_context-15', 'mwoz-recommend-0.4', 'mwoz-recommend-0.1']
total = 0
for file_name in output_file:
    print(f"gen: {file_name}:")
    res_list = []
    if 'context' in file_name:
        max_turn = file_name.split('-')[-1]
    else:
        max_turn = 15
    test_datas = json.load(open(f'data/test.soloist_filled_{max_turn}.json', 'r'))
    predictions = json.load(open('output_new/' + file_name + '.json', 'r'))
    bs = json.load(open('output_new/' + file_name + '_bs.json', 'r'))
    cache = {}
    current_name = 'SNG0073.json'
    pre_res = []
    gt_res_list = []
    pre_res_all = []
    gt_res_all = []
    for idx, pred in tqdm(enumerate(predictions)):
        test_data = test_datas[idx]
        file = test_data['name']
        gt_res = test_data["reply"]
        res, _ =  parse_decoding_results_direct(pred)
        if file != current_name:
            goal = parse_belief_state_all(parse_goal(data[current_name]['goal']))
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
            current_name = file
            cache = {}
            data_save_pred = {
                'context':pre_res,
                'success': success,
            }
            pre_res_all.append(data_save_pred)
            data_save_gt = {
                'context': gt_res_list,
                'success': success,
            }
            gt_res_all.append(data_save_gt)
            pre_res = []
            gt_res_list = []
        res_bs = bs[idx]
        for domain in res_bs.keys():
            if domain not in cache.keys():
                cache[domain] = res_bs[domain]
            elif len(res_bs[domain].keys()) > len(cache[domain].keys()):
                cache[domain] = res_bs[domain]
        pre_res.append(res)
        gt_res_list.append(gt_res)

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
    data_save_pred = {
        'context': pre_res,
        'success': success,
    }
    pre_res_all.append(data_save_pred)
    data_save_gt = {
        'context': gt_res_list,
        'success': success,
    }
    gt_res_all.append(data_save_gt)
    json.dump(pre_res_all, open('output_new/' + file_name + '_res.json', 'w'))
    json.dump(gt_res_all, open('output_new/gt_res.json', 'w'))
