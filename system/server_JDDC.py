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
from tokenizations.bpe_tokenizer import get_encoder
from collections import defaultdict
from nltk import word_tokenize
from accelerate import Accelerator


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
        self.tokenizer_path = 'JDDC_data/vocab.txt'
        self.special_tokens = 'JDDC_data/special_tokens.txt'
        self.eos_token = '<|endoftext|>'
        self.max_turn = 15
        self.segment = False
        self.bpe_token = False
        self.encoder_json = "tokenizations/encoder.json"
        self.vocab_bpe = "tokenizations/vocab.bpe"
        self.max_length = 128
        self.max_context_length = 90


args = Arg()

app = Flask(__name__)

# tokenizer
if args.segment:
    from tokenizations import tokenization_bert_word_level as tokenization_bert
else:
    from tokenizations import tokenization_bert
if args.bpe_token:
    tokenizer = get_encoder(args.encoder_json, args.vocab_bpe)
else:
    tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
for line in open(args.special_tokens, 'r', encoding='utf-8'):
    line = line.strip('\n')
    tokenizer.add_tokens(line)
tokenizer.add_tokens(args.eos_token)
tokenizer.max_len = args.max_length
tokenizer.bos_token = args.eos_token
tokenizer.eos_token = args.eos_token
tokenizer.unk_token = args.eos_token
tokenizer.padding_side = "left"

system_token_id = tokenizer.convert_tokens_to_ids(['系统：'])
user_token_id = tokenizer.convert_tokens_to_ids(['用户：'])
induction_id = tokenizer.convert_tokens_to_ids(['=>'])
total_all = 0.0
total = defaultdict(int)
success_num = defaultdict(int)
total_turn = defaultdict(int)
total_dialog = defaultdict(int)
system_path = 'system_JDDC/'
flag = True
system_assign = {
    'jddc-context-15': [i for i in range(6)] + [i for i in range(30, 36)],
    'jddc-context-3': [i for i in range(6, 12)],
    'jddc-context-1': [i for i in range(12, 18)],
    'jddc-domain-3': [i for i in range(18, 24)],
    'jddc-domain-1': [i for i in range(24, 30)],
}
# system_assign = {
#     'redial-context-15': [0, 1, 2],
#     'redial-context-3': [3, 4, 5],
#     'redial-context-1': [6, 7, 8],
#     'redial-domain-3': [9, 10, 11],
#     'redial-domain-1': [12, 13, 14],
#     'redial-recommend-3': [15, 16, 17],
#     'redial-recommend-1': [18, 19, 20],
#     # 'redial-context-15': [i for i in range(42, 48)],
#     # 'jddc-context-15': [i for i in range(48, 54)]
# }

system_assign_index = {
    'jddc-context-15': 0,
    'jddc-context-3': 0,
    'jddc-context-1': 0,
    'jddc-domain-3': 0,
    'jddc-domain-1': 0,
    # 'redial-context-15': 0,
    # 'jddc-context-15': 0,
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


@app.route('/', methods=['GET', 'POST'])
def generate_response():
    global total_all, system_token_id, user_token_id, total, success_num, total_turn, system_path, total_dialog, system_assign, system_assign_index
    system_type = request.form['system']
    task_type = request.form['task']
    cache = json.loads(request.form['cache'])

    args.max_turn = int(system_type.split('_')[-1].split('-')[-1])
    if args.max_turn > 10:
        args.max_length = 256
        args.max_context_length = 210
    if args.max_turn == 3:
        args.max_length = 128
        args.max_context_length = 90
    if args.max_turn == 1:
        args.max_length = 64
        args.max_context_length = 40
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
    if system_type.split('-')[0] == 'jddc':
        context = request.form['context']
        # logger.error(f"origin_context: {context}")
        # context = norm_text(context)
        # logger.error(f"norm_context: {context}")
        context = context.replace('用户 ：', '用户：').replace('系统 ：', '系统：').replace('【用户】', '用户：').replace('【系统】', '系统：').replace('用户：用户：', '用户：').replace('【再见】', '').replace('[UNK]', '')
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
            context = ''.join(context)
            source = context + '=>'
            encode_ans = tokenizer.encode_plus(source, truncation=False)
            encode_ans['input_ids'] = encode_ans['input_ids'][-args.max_context_length:]
            encode_ans['token_type_ids'] = encode_ans['token_type_ids'][-args.max_context_length:]
            logger.error(encode_ans)
            context_tokens = encode_ans['input_ids']
            token_type_ids = encode_ans['token_type_ids']

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
                        res = read_pkl(file_path)
                    except:
                        logger.error(f"error system_id: {number_system}")
                        continue
                    try:
                        os.remove(file_path)
                    except:
                        logger.error(f"fail to remove: {file_path}")
                    break

        if 'index' in request.form.keys():
            dialog_data['turn'] = len(dialog_data['context']) - 1
            dialog_data['state'] = copy.deepcopy(cache)
            dialog_data['system'] = system_type
            dialog_data['task'] = request.form['task']
            logger.error(dialog_data)
            turn_now = total[task_type]
            write_pkl(dialog_data, f'output/interaction/{system_type}_{task_type}_{turn_now}.json')
            total[task_type] += 1
            total_dialog[task_type] += dialog_data['turn']
            logging.error(f"total_{task_type}: {total[task_type]}")
            logging.error(f"total_dialog_{task_type}: {total_dialog[task_type]}")
            for k in total.keys():
                logging_txt = 'task: ' + k + ' turn: ' + str(int(total[k])) + \
                              ' avg_turn: ' + str(total_dialog[k] / total[k])
                logger.error(logging_txt)
            return ''
        res = '系统： ' + res
        logger.info(res)
        data_to_send = {
            'text': res,
            'cache': json.dumps(cache)
        }
        return json.dumps(data_to_send)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
