from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import os
import pickle
import random
import re
import time

import torch
from flask import Flask, request
from parse import parse_decoding_results_direct, parse_belief_state
from fill import fill, get_talk, get_item
from collections import defaultdict
from nltk import word_tokenize
# from time import time
def write_pkl(obj, filename):
    dirname = '/'.join(filename.split('/')[:-1])
    os.makedirs(dirname, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def read_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer
import faulthandler
# 在import之后直接添加以下启用代码即可
faulthandler.enable()
# 后边正常写你的代码

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('werkzeug')
logger.setLevel(logging.ERROR)

MAX_LENGTH = int(150)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer)
}


class Arg(object):
    def __init__(self):
        self.model_type = 'gpt2'
        self.model_name_or_path = 'model/all_15/final_tr0.3454456596468458/'
        self.tokenizer_path = 'model/pretrained/'
        self.special_tokens = 'data/special_tokens.txt'
        self.padding_text = ''
        self.length = 110
        self.num_samples = 5
        self.temperature = 1.0
        self.repetition_penalty = 1.0
        self.top_k = 0
        self.top_p = 0.5
        self.no_cuda = False
        self.seed = 42
        self.eos_token = '<|endoftext|>'
        self.max_turn = 15
        self.device = None

class Arg_redial(object):
    def __init__(self):
        self.tokenizer_path = 'pretrained_model/'
        self.special_tokens = 'redial_data/special_tokens.txt'
        self.eos_token = '<|endoftext|>'
        self.max_turn = 15


args = Arg()
args_redial = Arg_redial()

app = Flask(__name__)

def norm_text(text):
    text = text.lower()
    text = ' '.join(word_tokenize(text))
    text = text.replace('@ ', '@')
    text = text.replace(' :', ':')
    return text

_, tokenizer_class = MODEL_CLASSES[args.model_type]
tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path)
if args.special_tokens:
    for line in open(args.special_tokens, 'r', encoding='utf-8'):
        line = line.strip('\n')
        tokenizer.add_tokens(line)
tokenizer.eos_token = args.eos_token
tokenizer.pad_token = tokenizer.eos_token

# tokenizer
tokenizer_redial = GPT2Tokenizer.from_pretrained(args_redial.tokenizer_path)
if args_redial.special_tokens:
    for line in open(args_redial.special_tokens, 'r', encoding='utf-8'):
        line = line.strip('\n')
        tokenizer_redial.add_tokens(line)
tokenizer_redial.eos_token = args_redial.eos_token
tokenizer_redial.pad_token = tokenizer_redial.eos_token
tokenizer_redial.padding_side = 'left'

system_token_id = tokenizer.convert_tokens_to_ids(['system'])
user_token_id = tokenizer.convert_tokens_to_ids(['user'])
induction_id = tokenizer.convert_tokens_to_ids(['=>'])
total_all = 0.0
total = defaultdict(int)
success_num = defaultdict(int)
total_turn = defaultdict(int)
total_dialog = defaultdict(int)
system_path = 'system/'
flag = True
system_assign = {
    # 'mwoz-context-15': [i for i in range(6)] + [i for i in range(42, 54)],
    # 'mwoz-context-3': [i for i in range(6, 12)],
    # 'mwoz-context-1': [i for i in range(12, 18)],
    'mwoz-domain-3': [i for i in range(18)],
    # 'mwoz-domain-1': [i for i in range(24, 30)],
    # 'mwoz-recommend-3': [i for i in range(30, 36)],
    # 'mwoz-recommend-1': [i for i in range(36, 42)],
    # 'redial-context-15': [i for i in range(42, 48)],
    # 'JDDC-context-15': [i for i in range(48, 54)]
}
# system_assign = {
#     # 'mwoz-context-15': [0, 1, 2],
#     # 'mwoz-context-3': [3, 4, 5],
#     # 'mwoz-context-1': [6, 7, 8],
#     # 'mwoz-domain-3': [9, 10, 11],
#     # 'mwoz-domain-1': [12, 13, 14],
#     # 'mwoz-recommend-3': [15, 16, 17],
#     # 'mwoz-recommend-1': [18, 19, 20],
#     # 'redial-context-15': [i for i in range(42, 48)],
#     # 'JDDC-context-15': [i for i in range(48, 54)]
# }

system_assign_index = {
    'mwoz-context-15': 0,
    'mwoz-context-3': 0,
    'mwoz-context-1': 0,
    'mwoz-recommend-3': 0,
    'mwoz-recommend-1': 0,
    'mwoz-domain-3': 0,
    'mwoz-domain-1': 0,
    # 'redial-context-15': 0,
    # 'JDDC-context-15': 0,
}

def get_flag():
    global flag
    while True:
        if flag:
            flag = False
            break
        else:
            time.sleep(random.random() * 0.1)

def save_flag():
    global flag
    flag = True

def norm_text(text):
    # text = re.sub("/", " / ", text)
    # text = re.sub("\-", " \- ", text)
    text = re.sub("theres", "there's", text)
    text = re.sub("dont", "don't", text)
    text = re.sub("whats", "what's", text)
    text = text.lower()
    tokens = word_tokenize(text)
    text = ' '.join(tokens)
    return text

@app.route('/', methods=['GET', 'POST'])
def generate_response():
    global total_all, system_token_id, user_token_id, total, success_num, total_turn, system_path, total_dialog, system_assign, system_assign_index
    system_type = request.form['system']
    task_type = request.form['task']
    cache_data = json.loads(request.form['cache'])

    if "cache" not in cache_data.keys():
        cache_data["cache"]  ={}
    cache = cache_data["cache"]
    if "origin" not in cache_data.keys():
        cache_data["origin"] = []
    origin_context = cache_data["origin"]
    if "belief" not in cache_data.keys():
        cache_data["belief"] = []
    belief_state = cache_data["belief"]


    # input_dir = system_path + '60/input/'
    # output_dir = system_path + '60/output/'
    number_system = system_assign[system_type][system_assign_index[system_type]]
    logger.error(f"system_id: {number_system}")
    input_dir = system_path + str(number_system) + '/input/'
    output_dir = system_path + str(number_system) + '/output/'
    if system_assign_index[system_type] + 1 >= len(system_assign[system_type]):
        system_assign_index[system_type] = 0
    else:
        system_assign_index[system_type] += 1
    # if system_assign_index[system_type] + 1 >= len(system_assign[system_type]):
    #     system_assign_index[system_type] = 0
    # else:
    #     system_assign_index[system_type] += 1
    if 'clean' in request.form.keys():
        total_all -= total_turn[task_type]
        success_num[task_type] = 0
        total_turn[task_type] = 0
        total[task_type] = 0
        total_dialog[task_type] = 0
        return ''
    total_turn[task_type] += 1
    total_all += 1
    file_name_raw = system_type + '_' + str(int(total_all)) + '.json'
    file_path = input_dir + file_name_raw
    logger.error(task_type + ':' + str(total_turn[task_type]))
    if total_all % 50 == 0:
        logger.error(f'POST {total_all}')
    goal = request.form['goal']
    if system_type.split('-')[1] == 'context':
        max_turn = int(system_type.split('-')[-1])
    else:
        max_turn = 15
    if system_type.split('-')[0] == 'mwoz':
        goal = parse_belief_state(goal)
        context = request.form['context']
        # logger.error(f"origin_context: {context}")
        # context = norm_text(context)
        # logger.error(f"norm_context: {context}")
        context = context.replace('user:', 'user :')
        context = context.replace('system:', 'system :')
        context = context.split('[SEP]')

        # if system_type.split('-')[1] == 'recommend':
        #     recommend = int(system_type.split('-')[-1])
        # else:
        #     recommend = 15
        # turn = context[-1]
        logger.error(f"context {context}")
        if 'index' in request.form.keys():
            dialog_data = {}
            dialog_data['context'] = copy.copy(context)
            dialog_data['goal'] = request.form['goal']
            dialog_data['index'] = request.form['index']
        if 'index' not in request.form.keys():
            context = context[-max_turn:]
            context_ids = []
            token_ids_for_context = []
            for cxt in context:
                cxt = norm_text(cxt)
                ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cxt))
                context_ids += ids
                if 'user :' in cxt:
                    token_ids_for_context += user_token_id * len(ids)
                else:
                    token_ids_for_context += system_token_id * len(ids)
            response = '=>'
            response_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response))

            context_tokens = context_ids + response_id
            token_type_ids = token_ids_for_context + system_token_id
            data = {
                'context_tokens': context_tokens,
                'token_type_ids': token_type_ids
            }
            logger.error(f"input_data {data}")
            logger.error(f"file_path {file_path}")
            random_timestap = int(time.time())
            file_name = str(random_timestap) + "." + file_name_raw
            file_path = input_dir + file_name
            write_pkl(data, file_path)

            try_start_time = time.time()
            while True:
                time.sleep(0.1)
                try_end_time = time.time()
                try_time = try_end_time - try_start_time
                if try_time > 10:
                    try:
                        os.remove(file_path)
                    except:
                        logger.error(f"remove error {file_path}")
                        pass
                    try:
                        logger.error(f"error id {number_system}")
                        error_log_path = f"system/error_log.json"
                        error_log = read_pkl(error_log_path)
                        error_log.append(number_system)
                        write_pkl(error_log, error_log_path)
                    except:
                        pass
                    number_system = system_assign[system_type][system_assign_index[system_type]]
                    if system_assign_index[system_type] + 1 >= len(system_assign[system_type]):
                        system_assign_index[system_type] = 0
                    else:
                        system_assign_index[system_type] += 1
                    logger.error(f"Change to system_id: {number_system}")
                    input_dir = system_path + str(number_system) + '/input/'
                    output_dir = system_path + str(number_system) + '/output/'
                    random_timestap = int(time.time())
                    file_name = str(random_timestap) + "." + file_name_raw
                    file_path = input_dir + file_name
                    write_pkl(data, file_path)
                    try_start_time = time.time()
                    continue

                try:
                    file_list = os.listdir(output_dir)
                except:
                    time.sleep(0.1)
                    logger.error(f"error system_id: {number_system}")
                    continue

                if file_name in file_list:
                    time.sleep(0.1)
                    file_path = output_dir + file_name
                    logger.error(file_path)
                    try:
                        res, res_bs, _ = read_pkl(file_path)
                    except:
                        logger.error(f"error system_id: {number_system}")
                        continue
                    try:
                        os.remove(file_path)
                    except:
                        logger.error(f"fail to remove: {file_path}")
                    break

            # text_list = tokenizer.batch_decode(out, skip_special_tokens=True)
            # res, _ = parse_decoding_results_direct(text_list)
            belief_state.append(res_bs)

            for domain in res_bs.keys():
                if domain not in cache.keys():
                    cache[domain] = res_bs[domain]
                elif len(res_bs[domain].keys()) > len(cache[domain].keys()):
                    cache[domain] = res_bs[domain]

        if 'index' in request.form.keys():
            # logger.error(f"cache: {cache}")
            success = True
            get_flag()
            gt_items = get_item(goal)
            save_flag()
            get_flag()
            model_items = get_item(cache)
            save_flag()
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
            dialog_data['success'] = success
            dialog_data['turn'] = len(dialog_data['context']) - 1
            dialog_data['state'] = copy.deepcopy(cache)
            dialog_data['origin'] = copy.deepcopy(origin_context)
            dialog_data['belief'] = copy.deepcopy(belief_state)
            dialog_data['system'] = system_type
            dialog_data['task'] = request.form['task']
            logger.error(dialog_data)
            turn_now = total[task_type]
            write_pkl(dialog_data, f'output/interaction_train/{system_type}_{task_type}_{turn_now}.json')
            total[task_type] += 1
            success_num[task_type] += 1 if success else 0
            total_dialog[task_type] += dialog_data['turn']
            logging.error(f"total_{task_type}: {total[task_type]}")
            logging.error(f"total_dialog_{task_type}: {total_dialog[task_type]}")
            logging.error(f"success_num_{task_type}: {success_num[task_type]}")
            for k in total.keys():
                logging_txt = 'task: ' + k + ' turn: ' + str(int(total[k])) + \
                              ' success: ' + str(success_num[k] / total[k]) + ' avg_turn: ' + str(total_dialog[k] / total[k])
                logger.error(logging_txt)
            return ''
        talk = get_talk(res_bs, res)
        get_flag()
        res_origin = copy.deepcopy(res)
        res_fill = fill(res, res_bs, talk)
        save_flag()
        res_origin = 'system: ' + res_origin
        res = 'system: ' + res_fill
        logger.info(res_origin)
        logger.info(res)
        origin_context.append(res_origin)
        # logger.error(cache)

        cache_data = {
            "cache": cache,
            "origin": origin_context,
            "belief": belief_state,
        }
        data_to_send = {
            'text': res,
            'cache': json.dumps(cache_data)
        }
        return json.dumps(data_to_send)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
