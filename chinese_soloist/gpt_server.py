import os
import pickle
import time

import torch
from transformers import (
    set_seed,
    GPT2Config,
    GPT2LMHeadModel,
)
import json
import argparse
from torch.utils.data import Dataset, DataLoader
from tokenizations.bpe_tokenizer import get_encoder
from accelerate import Accelerator
import logging

def write_pkl(obj, filename):
    dirname = '/'.join(filename.split('/')[:-1])
    os.makedirs(dirname, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def read_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def test(args):
    accelerator = Accelerator(fp16=args.fp16, cpu=args.cpu, device_placement=True)
    set_seed(args.seed)

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
    accelerator.print('Tokenizer vocab_size is {}'.format(tokenizer.vocab_size))

    # model
    model_config = GPT2Config.from_json_file(args.model_config)
    model_config.vocab_size = tokenizer.vocab_size + 1000
    accelerator.print('Model config: ' + model_config.to_json_string())
    if not args.model_path:
        model = GPT2LMHeadModel(config=model_config)
    else:
        model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.resize_token_embeddings(tokenizer.vocab_size + 1000)

    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    accelerator.print('Number of parameters: {}'.format(num_parameters))
    model = accelerator.prepare(
        model
    )

    input_dir = args.system_file_path + 'input/'
    output_dir = args.system_file_path + 'output/'
    accelerator.print(input_dir)


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
                    context_tokens = context_tokens[-args.max_context_length:]
                    token_type_ids = token_type_ids[-args.max_context_length:]
                    context_tokens = torch.tensor(context_tokens, dtype=torch.long, device='cuda:0').unsqueeze(0)
                    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device='cuda:0').unsqueeze(0)
                    out = model.generate(
                        input_ids=context_tokens,
                        token_type_ids=token_type_ids,
                        max_length=args.max_length,
                        num_return_sequences=args.num_samples,
                        temperature=args.temperature,
                        repetition_penalty=args.repetition_penalty,
                        top_k=args.top_k,
                        top_p=args.top_p
                    )
                    decode_ans = tokenizer.batch_decode(out, skip_special_tokens=True)
                    for ans in decode_ans:
                        try:
                            res = ans.split('=>')[1].split('系统：')[1]
                        except:
                            res = ''
                        res = res.replace(' ', '')
                        break
                    file_path = output_dir + file
                    write_pkl(res, file_path)
                    accelerator.print(f'Done {file_path}')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_data_file", default='data/test_set.txt', type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default='output/', type=str,
                        help="save file.")

    # model parameters
    parser.add_argument('--model_config', default='pretrained_model/config.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument("--model_path", default='model/data-100_context-1/final_tr0.37055231996629845/', type=str,
                        help="The model checkpoint for weights initialization.")

    # decode parameters
    parser.add_argument("--max_length", default=128, type=int, help="")
    parser.add_argument("--max_context_length", default=90, type=int, help="")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--max_turn', type=int, default=1, help="number of turns used as context")
    parser.add_argument('--batch_size', type=int, default=400, help="number of batch size")

    # tokenizer parameters
    parser.add_argument('--tokenizer_path', default='pretrained_model/vocab.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--eos_token', default='<|endoftext|>', type=str, required=False, help='eos_token')
    parser.add_argument('--special_tokens', default='pretrained_model/special_tokens.txt', type=str, required=False,
                        help='special_tokens')
    parser.add_argument('--segment', default=False, action='store_true', help='中文以词为单位')
    parser.add_argument('--bpe_token', default=False, action='store_true', help='subword')
    parser.add_argument('--encoder_json', default="tokenizations/encoder.json", type=str, help="encoder.json")
    parser.add_argument('--vocab_bpe', default="tokenizations/vocab.bpe", type=str, help="vocab.bpe")

    # accelerator parameters
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument("--process_id", default=0, type=int)

    args = parser.parse_args()
    process_id = args.process_id
    system_file_path = '../multiwoz/system_JDDC/' + str(process_id) + '/'
    data_set = ''
    system = ''
    model = None

    config = json.load(open(system_file_path + 'config.json', 'r'))
    data_set = config['dataset']
    system = config['system']
    task = config['task']

    model_path = 'model/' + system + '/'
    for file in os.listdir(model_path):
        if file.startswith('final'):
            model_name = file
            break
    args.system_file_path = system_file_path
    args.model_path = 'model/' + system + '/' + model_name + '/'
    args.max_turn = int(system.split('_')[-1].split('-')[-1])
    if args.max_turn > 10:
        args.max_length = 256
        args.max_context_length = 210
    if args.max_turn == 3:
        args.max_length = 128
        args.max_context_length = 90
    if args.max_turn == 1:
        args.max_length = 64
        args.max_context_length = 40

    test(args)


if __name__ == '__main__':
    main()
