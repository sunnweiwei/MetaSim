from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import copy
import logging
import random

from tqdm import tqdm
import json

import torch
import torch.nn.functional as F
import numpy as np

from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(150)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer)
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


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


# def get_droped_bs(bs, num_drop=0.5):
#     replace_bs = {}
#     remove_bs = {}
#     for domain in bs.keys():
#         replace_bs[domain] = {}
#         num_slots = len(bs[domain].keys())
#         bs_list = list(bs[domain].keys())
#         random.shuffle(bs_list)
#         num_remain_slots = int(num_slots * num_drop)
#         if num_slots - num_remain_slots > 0:
#             remove_bs[domain] = {}
#         for index, slot in enumerate(bs_list):
#             if index < num_remain_slots:
#                 if slot in slot_value[domain].keys():
#                     if len(slot_value[domain][slot]) == 1:
#                         replace_bs[domain][slot] = bs[domain][slot]
#                     else:
#                         while True:
#                             value = random.choice(slot_value)
#                             if value != bs[domain][slot]:
#                                 replace_bs[domain][slot] =value
#                             else:
#                                 continue
#                 else:
#                     replace_bs[domain][slot] = bs[domain][slot]
#             else:
#                 replace_bs[domain][slot] = bs[domain][slot]
#                 remove_bs[domain][slot] = bs[domain][slot]
#         if num_slots % 2 == 1:
#             slot = bs_list[-1]
#             if len(slot_value[domain][slot]) == 1:
#                 replace_bs[domain][slot] = bs[domain][slot]
#             else:
#                 while True:
#                     value = random.choice(slot_value)
#                     if value != bs[domain][slot]:
#                         replace_bs[domain][slot] = value
#                     else:
#                         continue
#             remove_bs[domain].pop(slot)
#
#     return replace_bs, remove_bs

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


from parse import parse_decoding_results_direct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='gpt2', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='newest_model/data-100_context-1/final_tr0.19952457707954777/',
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
    parser.add_argument('--input_file', type=str, default='data/test.soloist_filled_3.json',
                        help="input json file to decoding")
    parser.add_argument('--output_dir', type=str, default='output_new/', help="save path")
    parser.add_argument('--max_turn', type=int, default=1, help="number of turns used as context")
    parser.add_argument('--batch_size', type=int, default=8, help="number of batch size")
    parser.add_argument('--num_mask', type=float, default=1, help="number of batch size")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path)
    if args.special_tokens:
        for line in open(args.special_tokens, 'r', encoding='utf-8'):
            line = line.strip('\n')
            tokenizer.add_tokens(line)
    tokenizer.eos_token = args.eos_token
    tokenizer.pad_token = tokenizer.eos_token
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    logger.info(args)

    system_token_id = tokenizer.convert_tokens_to_ids(['system'])
    user_token_id = tokenizer.convert_tokens_to_ids(['user'])
    induction_id = tokenizer.convert_tokens_to_ids(['=>'])

    logger.info(f'Start decode {args.input_file}...')
    inputs = json.load(open(args.input_file))
    output_tests = []
    bs_list = []

    for idx in tqdm(range(0, len(inputs), args.batch_size), desc=f'inputs'):
        examps = inputs[idx: idx + args.batch_size]
        context_tokens = []
        token_type_ids = []
        attention_masks = []
        length = 0
        for examp in examps:
            history = examp['history']
            context = history[-args.max_turn:]
            context_ids = []
            token_ids_for_context = []
            attention_mask = []
            for cxt in context:
                ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cxt))
                context_ids += ids
                attention_mask += [1] * len(ids)
                if 'user :' in cxt:
                    token_ids_for_context += user_token_id * len(ids)
                else:
                    token_ids_for_context += system_token_id * len(ids)

            context_token = context_ids + induction_id
            token_type_id = token_ids_for_context + system_token_id
            attention_mask = attention_mask + [1]
            length = len(context_token) if len(context_token) > length else length
            context_tokens.append(context_token)
            token_type_ids.append(token_type_id)
            attention_masks.append(attention_mask)

        for token_idx in range(len(context_tokens)):
            context_token = context_tokens[token_idx]
            if len(context_token) < length:
                token_type_ids[token_idx] = [0] * (length - len(context_token)) + token_type_ids[token_idx]
                attention_masks[token_idx] = [0] * (length - len(context_token)) + attention_masks[token_idx]
                context_tokens[token_idx] = [tokenizer.pad_token_id] * (length - len(context_token)) + context_tokens[
                    token_idx]

        for token_idx in range(len(context_tokens)):
            token_type_ids[token_idx] = token_type_ids[token_idx][-args.length:]
            attention_masks[token_idx] = attention_masks[token_idx][-args.length:]
            context_tokens[token_idx] = context_tokens[token_idx][-args.length:]

        def generate_sequence(model, length, context, token_type_ids, attention_mask, num_samples=1, temperature=1.0,
                              top_k=0,
                              top_p=0.0, repetition_penalty=1.0, device='cpu', num_mask=1):
            context_ori = copy.deepcopy(context)
            # print(f'context_ori: {context_ori}')
            context = torch.tensor(context, dtype=torch.long, device=device)
            token_type_ids_ori = copy.deepcopy(token_type_ids)
            # print(f'token_type_ids_ori: {token_type_ids_ori}')
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=device)
            token_type_ids = token_type_ids.repeat(num_samples, 1)
            attention_mask_ori = copy.deepcopy(attention_mask)
            # print(f'attention_mask_ori: {attention_mask_ori}')
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
            # print(f'text_list:{text_list}')
            length = 0
            remove_bs_list = []
            for idx in range(len(context_ori)):
                res, res_bs = parse_decoding_results_direct(text_list[idx*args.num_samples : (idx+1)*args.num_samples])
                # # logger.info(f"res: {res}")
                # # logger.info(f'origin_bs: {res_bs}')
                replace_bs, remove_bs = get_droped_bs(res_bs, num_mask)
                remove_bs_list.append(remove_bs)
                # # logger.info(f'replace: {replace_bs}')
                # # logger.info(f'remove: {remove_bs}')
                bs_str = convert_to_str(replace_bs)
                bs_str = bs_str.strip(' ;').strip(';').replace(';|', '|')
                bs_str += 'system :'
                bs_str_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(bs_str))
                token_type_ids_ori[idx] += system_token_id * len(bs_str_token)
                context_ori[idx] += bs_str_token
                attention_mask_ori[idx] += [1] * len(bs_str_token)
                length = len(token_type_ids_ori[idx]) if len(token_type_ids_ori[idx]) > length else length
                # context_ori = torch.tensor(context_ori, dtype=torch.long, device=device).unsqueeze(0)
                # token_type_ids_ori = torch.tensor(token_type_ids_ori, dtype=torch.long, device=device).unsqueeze(0)

            for token_idx in range(len(context_ori)):
                context_token = context_ori[token_idx]
                if len(context_token) < length:
                    token_type_ids_ori[token_idx] = [0] * (length - len(context_token)) + token_type_ids_ori[token_idx]
                    attention_mask_ori[token_idx] = [0] * (length - len(context_token)) + attention_mask_ori[token_idx]
                    context_ori[token_idx] = [tokenizer.pad_token_id] * (length - len(context_token)) + \
                                                context_ori[
                                                    token_idx]

            for token_idx in range(len(context_ori)):
                token_type_ids_ori[token_idx] = token_type_ids_ori[token_idx][-args.length:]
                attention_mask_ori[token_idx] = attention_mask_ori[token_idx][-args.length:]
                context_ori[token_idx] = context_ori[token_idx][-args.length:]

            # logger.error(tokenizer.batch_decode(context_ori, skip_special_tokens=True))
            # logger.error(token_type_ids_ori)
            # logger.error(attention_mask_ori)
            context_ori = torch.tensor(context_ori, dtype=torch.long, device=device)
            token_type_ids_ori = torch.tensor(token_type_ids_ori, dtype=torch.long, device=device)
            token_type_ids_ori = token_type_ids_ori.repeat(num_samples, 1)
            attention_mask_ori = torch.tensor(attention_mask_ori, dtype=torch.long, device=device)
            attention_mask_ori = attention_mask_ori.repeat(num_samples, 1)
            generated = context_ori
            generated = model.generate(input_ids=generated,
                                       token_type_ids=token_type_ids_ori,
                                       attention_mask=attention_mask_ori,
                                       max_length=args.max_length,
                                       temperature=temperature,
                                       top_k=top_k,
                                       do_sample=True,
                                       num_return_sequences=num_samples,
                                       repetition_penalty=repetition_penalty,
                                       top_p=top_p,
                                       )
            return generated, remove_bs_list

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
    model_str = args.model_name_or_path.split('/')[1]
    if args.num_mask != 1:
        model_str = f'mwoz-recommend-{args.num_mask}'
    json.dump(output_tests, open(args.output_dir + f'{model_str}.json', 'w'), indent=2)
    json.dump(bs_list, open(args.output_dir + f'{model_str}_bs.json', 'w'), indent=4)


if __name__ == '__main__':
    main()
