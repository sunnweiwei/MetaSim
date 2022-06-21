from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import copy
import logging
import os
import random
import re

from tqdm import tqdm
import json

import torch
import torch.nn.functional as F
import numpy as np

from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer
from transformers import set_seed

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(150)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer)
}

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

import random
import copy

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

def eval_slot_match(predicts, answers, skip_null=True):
    match = []
    for pred, ans in zip(predicts, answers):
        if len(ans) == 0 and skip_null:
            continue
        match.append(int(all([slot in pred for slot in ans])))
    return {'slot_match': sum(match) / len(match) * 100}


from parse import parse_decoding_results_direct
from fill import get_item
from parse import parse_belief_state_all
from evaluation import eval_all
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='gpt2', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='retrain_model/data-100_context-15_clean/epoch20_0_tr0.04692611491617358/',
                        type=str,
                        help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument('--tokenizer_path', default='model/pretrained/', type=str, required=False, help='选择词库')
    parser.add_argument('--special_tokens', default='data/special_tokens.txt', type=str, required=False,
                        help='special_tokens')
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--xlm_lang", type=str, default="", help="Optional language when used with the XLM model.")
    parser.add_argument("--length", type=int, default=386)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--eos_token', type=str, default='<|endoftext|>',
                        help="Token at which text generation is stopped")
    parser.add_argument('--input_file', type=str, default='data/test.soloist_filled_15.json',
                        help="input json file to decoding")
    parser.add_argument('--output_dir', type=str, default='retrain_output/', help="save path")
    parser.add_argument('--max_turn', type=int, default=15, help="number of turns used as context")
    parser.add_argument('--batch_size', type=int, default=16, help="number of batch size")
    parser.add_argument('--num_mask', type=float, default=1, help="number of batch size")

    # accelerate parameters
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    model_str = args.model_name_or_path.split('/')[-2].split('_')[0]

    accelerator = Accelerator(fp16=args.fp16, cpu=args.cpu, device_placement=True)
    set_seed(args.seed)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path)
    if args.special_tokens:
        for line in open(args.special_tokens, 'r', encoding='utf-8'):
            line = line.strip('\n')
            tokenizer.add_tokens(line)
    tokenizer.eos_token = args.eos_token
    tokenizer.pad_token = tokenizer.eos_token
    # model
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    if args.special_tokens:
        model.resize_token_embeddings(len(tokenizer))
    accelerator.print('Model config: ' + model.config.to_json_string())

    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    accelerator.print('Number of parameters: {}'.format(num_parameters))
    model.eval()

    system_token_id = tokenizer.convert_tokens_to_ids(['system'])
    user_token_id = tokenizer.convert_tokens_to_ids(['user'])
    induction_token_id = tokenizer.convert_tokens_to_ids(['=>'])

    # Data loader
    class MyDataset(Dataset):
        def __init__(self):
            self.examples = []
            self.token_ids = []
            self.attention_masks = []

            examples = json.load(open(args.input_file, 'r'))

            for examp in examples:
                history = examp['history']
                context = history[-args.max_turn:]
                context_ids = []
                token_ids_for_context = []
                for cxt in context:
                    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cxt))
                    context_ids += ids
                    if 'user :' in cxt:
                        token_ids_for_context += user_token_id * len(ids)
                    else:
                        token_ids_for_context += system_token_id * len(ids)

                context_token = context_ids + induction_token_id
                token_type_id = token_ids_for_context + system_token_id

                if len(context_token) < args.length:
                    attention_mask = [0] * args.length
                    attention_mask[:len(context_token)] = [1] * len(context_token)
                    context_token += [0] * (args.length - len(context_token))
                    token_type_id += [0] * (args.length - len(token_type_id))
                else:
                    attention_mask = [1] * args.length
                    context_token = context_token[-args.length:]
                    token_type_id = token_type_id[-args.length:]

                self.examples.append(context_token)
                self.token_ids.append(token_type_id)
                self.attention_masks.append(attention_mask)

            accelerator.print(f'Total examples is {len(self.examples)}')

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, item):
            return torch.tensor(self.examples[item]), torch.tensor(self.token_ids[item]), torch.tensor(self.attention_masks[item]),

    test_dataset = MyDataset()
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
    )
    model, test_loader = accelerator.prepare(
        model, test_loader
    )

    logger.info(f'Start decode {args.input_file}...')
    output_tests = []
    bs_list = []

    def generate_sequence(model, length, context, token_type_ids, attention_mask, num_samples=1, temperature=1.0,
                          top_k=0,
                          top_p=0.0, repetition_penalty=1.0, device='cpu', num_mask=1):
        context_ori = copy.deepcopy(context)
        context = torch.tensor(context, dtype=torch.long, device=device)
        token_type_ids_ori = copy.deepcopy(token_type_ids)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=device)
        token_type_ids = token_type_ids.repeat(num_samples, 1)
        attention_mask_ori = copy.deepcopy(attention_mask)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
        attention_mask = attention_mask.repeat(num_samples, 1)
        generated = context
        generated = model.generate(input_ids=generated,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask,
                                   max_length=args.max_length,
                                   temperature=temperature,
                                   top_k=top_k,
                                   do_sample=True,
                                   num_return_sequences=num_samples,
                                   repetition_penalty=repetition_penalty,
                                   top_p=top_p,
                                   )
        text_list = tokenizer.batch_decode(generated, skip_special_tokens=True)
        remove_bs_list = []
        for idx in range(len(context_ori)):
            res, res_bs = parse_decoding_results_direct(text_list[idx * args.num_samples: (idx + 1) * args.num_samples])
            replace_bs, remove_bs = get_droped_bs(res_bs, num_mask)
            remove_bs_list.append(remove_bs)
        return generated, remove_bs_list

    tk = tqdm(enumerate(test_loader), total=len(test_loader), desc='batch')
    for step, batch in tk:
        context_tokens, token_type_ids, attention_masks = batch
        out, bs = generate_sequence(
            model=model,
            context=context_tokens,
            token_type_ids=token_type_ids,
            attention_mask=attention_masks,
            num_samples=args.num_samples,
            length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=args.device,
            num_mask=args.num_mask,
        )
        text_list = tokenizer.batch_decode(out, skip_special_tokens=True)
        count = 0
        examples = []
        for text in text_list:
            examples.append(text)
            count += 1
            if count == args.num_samples:
                output_tests.append(examples)
                examples = []
                count = 0
        bs_list.extend(bs)
    json.dump(output_tests, open(args.output_dir + f'{model_str}_clean.json', 'w'), indent=2)
    json.dump(bs_list, open(args.output_dir + f'{model_str}_clean_bs.json', 'w'), indent=4)

    ## eval
    gt = json.load(open('output/gt.json', 'r'))
    data = json.load(open('data/multi-woz/data.json', 'r'))

    total = 0
    num_success = 0
    max_turn = 15
    test_datas = json.load(open(f'data/test.soloist_filled_{max_turn}.json', 'r'))
    predictions = output_tests
    bs = bs_list
    # slot
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
        res, _ = parse_decoding_results_direct(pred)
        pre_res.append(res)
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
    total += 1
    if success:
        num_success += 1
    print(f'success: {num_success / total}')
    print(f"slot: {eval_slot_match(pre_res, slot_ans)}")
    print(eval_all(pre_res, gt))


if __name__ == '__main__':
    main()
