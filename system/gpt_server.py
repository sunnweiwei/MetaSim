import argparse
import copy
import json
import os
import pickle
import random
import time

import torch
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer

def write_pkl(obj, filename):
    dirname = '/'.join(filename.split('/')[:-1])
    os.makedirs(dirname, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def read_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--process_id", default=0, type=int)
    process_id = parser.parse_args().process_id

    system_file_path = 'system/' + str(process_id) + '/'
    data_set = ''
    system = ''
    model = None

    config = json.load(open(system_file_path + 'config.json', 'r'))
    data_set = config['dataset']
    system = config['system']
    task = config['task']

    model_path = 'newest_model/' + system + '/'
    if system == 'data-100_context-15':
        model_name = 'epoch15_20000_tr0.3656222161591053'
    else:
        for file in os.listdir(model_path):
            if file.startswith('final'):
                model_name = file
                break


    class Arg(object):
        def __init__(self):
            self.model_name_or_path = 'newest_model/' + system + '/' + model_name + '/'
            self.model_type = 'gpt2'
            self.tokenizer_path = 'model/pretrained/'
            self.special_tokens = 'data/special_tokens.txt'
            self.eos_token = '<|endoftext|>'
            self.padding_text = ''
            self.max_length = 512
            self.length = 470
            self.num_samples = 5
            self.temperature = 1.0
            self.repetition_penalty = 1.0
            self.top_k = 0
            self.top_p = 0.5
            self.seed = 42
            self.max_turn = int(system.split('_')[-1].split('-')[-1])
            self.num_mask = 1


    args = Arg()

    if args.max_turn == 15:
        args.max_length = 512
        args.length = 384
    else:
        args.max_length = 256
        args.length = 210

    if task.split('-')[1] == 'recommend':
        if int(task.split('-')[-1]) == 3:
            args.num_mask = 0.4
        elif int(task.split('-')[-1]) == 1:
            args.num_mask = 0.1

    MODEL_CLASSES = {
        'gpt2': (GPT2LMHeadModel)
    }

    def set_seed(args):
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)


    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
    if args.special_tokens:
        for line in open(args.special_tokens, 'r', encoding='utf-8'):
            line = line.strip('\n')
            tokenizer.add_tokens(line)
    tokenizer.eos_token = args.eos_token
    tokenizer.pad_token = tokenizer.eos_token

    system_token_id = tokenizer.convert_tokens_to_ids(['system'])
    user_token_id = tokenizer.convert_tokens_to_ids(['user'])
    induction_id = tokenizer.convert_tokens_to_ids(['=>'])

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args)

    args.model_type = args.model_type.lower()
    model_class = MODEL_CLASSES[args.model_type]
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    domain_attr = {
        'attraction': ['id', 'address', 'area', 'entrance fee', 'name', 'phone', 'postcode', 'pricerange', 'openhours',
                       'type'],
        'restaurant': ['id', 'address', 'area', 'food', 'introduction', 'name', 'phone', 'postcode', 'pricerange',
                       'location', 'type'],
        'hotel': ['id', 'address', 'area', 'internet', 'parking', 'single', 'double', 'family', 'name', 'phone',
                  'postcode', 'pricerange', 'takesbookings', 'stars', 'type'],
        'train': ['id', 'arriveBy', 'day', 'departure', 'destination', 'duration', 'leaveAt', 'price'],
        'hospital': ['department', 'id', 'phone'],
    }
    from evaluate import MultiWozDB
    from collections import defaultdict

    db = MultiWozDB()
    slot_value = defaultdict(dict)
    for domain in domain_attr.keys():
        for slot in domain_attr[domain]:
            slot_value[domain][slot] = []
    for domain in domain_attr.keys():
        venues = db.queryResultVenues(domain, bs={})
        for venue in venues:
            for id in range(len(domain_attr[domain])):
                if venue[id] not in slot_value[domain][domain_attr[domain][id]]:
                    slot_value[domain][domain_attr[domain][id]].append(venue[id])
    # print(slot_value)
    slot_value['police'] = {
        "name": ["New York Police Station"],
        "address": ["New York"],
        "id": [2],
        "phone": ["1777726362"],
        "postcode": ["cb14dp"]
    }
    slot_value['taxi'] = {
        "type": ["Benz"],
        "phone": ["111111111"],
        "time": ["2:30"],
    }

    from parse import parse_decoding_results_direct


    def get_droped_bs(bs, num_drop=0.5):
        remove_bs = {}
        replace_bs = copy.deepcopy(bs)
        for domain in bs.keys():
            remove_bs[domain] = {}
            num_slots = len(bs[domain].keys())
            bs_list = list(bs[domain].keys())
            random.shuffle(bs_list)
            num_remain_slots = int(num_slots * num_drop + random.random())
            for slot in bs_list[:num_remain_slots]:
                remove_bs[domain][slot] = bs[domain][slot]
            for slot in bs_list[num_remain_slots:]:
                if slot not in slot_value[domain] or len(slot_value[domain][slot]) < 2:
                    continue
                while replace_bs[domain][slot] == bs[domain][slot]:
                    replace_bs[domain][slot] = random.choice(slot_value[domain][slot])  # replace
            # if len(remove_bs[domain]) == 0:
            #     remove_bs.pop(domain)
        return replace_bs, remove_bs

    def convert_to_str(state):
        state_str = 'belief :'
        first_domain = True
        for domain in state.keys():
            if first_domain:
                state_str += ' ' + domain
                first_domain = False
            else:
                state_str += '| ' + domain
            for slot in state[domain].keys():
                state_str = state_str + ' ' + str(slot) + ' = ' + str(state[domain][slot]) + ' ;'
        # print(state_str)
        return state_str


    def write_pkl(obj, filename):
        dirname = '/'.join(filename.split('/')[:-1])
        os.makedirs(dirname, exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)

    def generate_sequence(model, length, context, token_type_ids, num_samples=1, temperature=1.0, top_k=0,
                          top_p=0.0, repetition_penalty=1.0, device='cpu', num_mask=1):
        context_ori = copy.deepcopy(context)
        # logger.error(f'origin: {tokenizer.batch_decode(context_ori)}')
        context = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        token_type_ids_ori = copy.deepcopy(token_type_ids)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=device).unsqueeze(0)
        text_list = []
        try_name = 0
        while len(text_list) == 0:
            generated = model.generate(input_ids=context,
                                       token_type_ids=token_type_ids,
                                       max_length=args.max_length,
                                       temperature=temperature,
                                       top_k=top_k,
                                       do_sample=True,
                                       num_return_sequences=num_samples,
                                       repetition_penalty=repetition_penalty,
                                       top_p=top_p,
                                       )
            text_list = tokenizer.batch_decode(generated, skip_special_tokens=True)
            if try_name < 5:
                text_list = [txt for txt in text_list if '!!!!' not in txt]
            else:
                text_list = [txt.replace('!', '') for txt in text_list]
                error_log = dict(input_ids=context,
                                 token_type_ids=token_type_ids,
                                 max_length=args.max_length,
                                 temperature=temperature,
                                 top_k=top_k,
                                 do_sample=True,
                                 num_return_sequences=num_samples,
                                 repetition_penalty=repetition_penalty,
                                 top_p=top_p,
                                 )
                import string
                output_file = f"output/error_log_{''.join(random.sample(string.ascii_letters + string.digits, 8))}"
                write_pkl(error_log, output_file)
                logger.error(output_file)
                break
            try_name += 1

        res, res_bs = parse_decoding_results_direct(text_list)
        if num_mask > 0.9:
            return generated, res, res_bs
        # logger.info(f"res: {res}")
        # logger.info(f'origin_bs: {res_bs}')
        replace_bs, remove_bs = get_droped_bs(res_bs, num_mask)
        # logger.info(f'replace: {replace_bs}')
        # logger.info(f'remove: {remove_bs}')
        bs_str = convert_to_str(replace_bs)
        bs_str = bs_str.strip(' ;').strip(';').replace(';|', '|')
        bs_str += 'system : '
        bs_str_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(bs_str))
        token_type_ids_ori += system_token_id * len(bs_str_token)
        context_ori += bs_str_token
        token_type_ids_ori = token_type_ids_ori[-args.length:]
        context_ori = context_ori[-args.length:]
        # logger.error(tokenizer.batch_decode(context_ori, skip_special_tokens=True))
        context_ori = torch.tensor(context_ori, dtype=torch.long, device=device).unsqueeze(0)
        token_type_ids_ori = torch.tensor(token_type_ids_ori, dtype=torch.long, device=device).unsqueeze(0)
        text_list = []
        try_name = 0
        while len(text_list) == 0:
            generated = model.generate(input_ids=context_ori,
                                       token_type_ids=token_type_ids_ori,
                                       max_length=args.max_length,
                                       temperature=temperature,
                                       top_k=top_k,
                                       do_sample=True,
                                       num_return_sequences=num_samples,
                                       repetition_penalty=repetition_penalty,
                                       top_p=top_p,
                                       )
            text_list = tokenizer.batch_decode(generated, skip_special_tokens=True)
            if try_name < 5:
                text_list = [txt for txt in text_list if '!!!!' not in txt]
            else:
                text_list = [txt.replace('!', '') for txt in text_list]
                error_log = dict(input_ids=context_ori,
                                       token_type_ids=token_type_ids_ori,
                                       max_length=args.max_length,
                                       temperature=temperature,
                                       top_k=top_k,
                                       do_sample=True,
                                       num_return_sequences=num_samples,
                                       repetition_penalty=repetition_penalty,
                                       top_p=top_p,
                                 )
                import string
                output_file = f"output/error_log_{''.join(random.sample(string.ascii_letters + string.digits, 8))}"
                write_pkl(error_log, output_file)
                logger.error(output_file)
                break
            try_name += 1
        res, res_bs = parse_decoding_results_direct(text_list)
        return generated, res, remove_bs


    input_dir = system_file_path + 'input/'
    output_dir = system_file_path + 'output/'
    if task.split('-')[1] == 'recommend':
        num_mask = int(task.split('-')[-1])
        if num_mask == 3:
            num_mask = 0.5
        else:
            num_mask = 0
    else:
        num_mask = 1
    print(task)

    while True:
        file_list = os.listdir(input_dir)
        if file_list:
            file_list.sort(key=lambda x: -int(x[:x.index('.')]))
            file_list = [file_list[0]]
            for file in file_list:
                if file:
                    time.sleep(0.01)
                    file_path = input_dir + file
                    try:
                        data = read_pkl(file_path)
                        os.remove(file_path)
                    except:
                        break
                    context_tokens = data['context_tokens']
                    token_type_ids = data['token_type_ids']
                    context_tokens = context_tokens[-args.length:]
                    token_type_ids = token_type_ids[-args.length:]
                    out, res, bs = generate_sequence(
                        model=model,
                        context=context_tokens,
                        token_type_ids = token_type_ids,
                        num_samples=args.num_samples,
                        length=args.max_length,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                        device=args.device,
                        num_mask=num_mask,
                    )
                    out = out.tolist()
                    file_path = output_dir + file
                    file_name = file.split('.')[0]
                    write_pkl([res, bs, out], output_dir + file)
                    print(f"Done {file}")
