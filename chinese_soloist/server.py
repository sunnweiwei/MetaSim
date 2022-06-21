import os

import torch
from transformers import (
    set_seed,
    GPT2Config,
    GPT2LMHeadModel,
)
import json
import argparse
from accelerate import Accelerator

def test(args):
    accelerator = Accelerator(fp16=args.fp16, cpu=args.cpu, device_placement=True)
    set_seed(args.seed)

    # model
    if not args.model_path:
        model_config = GPT2Config.from_json_file(args.model_config)
        model = GPT2LMHeadModel(config=model_config)
    else:
        model = GPT2LMHeadModel.from_pretrained(args.model_path)
    accelerator.print('Model config: ' + model.config.to_json_string())

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
    while True:
        file_list = os.listdir(input_dir)
        if file_list:
            for file in file_list:
                if file:
                    file_path = input_dir + file
                    try:
                        data = json.load(open(file_path, 'r'))
                    except:
                        break
                    os.remove(file_path)
                    context_tokens = data['context_tokens']
                    context_tokens = torch.tensor(context_tokens, dtype=torch.long, device='cuda:0').unsqueeze(0)
                    accelerator.print(context_tokens)
                    out = model.generate(
                            input_ids=context_tokens,
                            max_length=args.max_length,
                            do_sample=True,
                            num_return_sequences=args.num_samples,
                            temperature=args.temperature,
                            repetition_penalty=args.repetition_penalty,
                            top_k=args.top_k,
                            top_p=args.top_p
                        )
                    out = out.tolist()
                    file_path = output_dir + file
                    json.dump(out, open(file_path, 'w'))
                    accelerator.print(f'Done {file}')

def main():
    parser = argparse.ArgumentParser()

    ## Other parameters
    # model parameters
    parser.add_argument("--test_data_file", default='data/test_set.txt', type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--save_file", default='decode.txt', type=str,
                        help="save file.")

    # model parameters
    parser.add_argument('--model_config', default='pretrained_model/config.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument("--model_path", default='model/final', type=str,
                        help="The model checkpoint for weights initialization.")

    # decode parameters
    parser.add_argument("--max_length", default=256, type=int, help="")
    parser.add_argument("--max_context_length", default=220, type=int, help="")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--max_turn', type=int, default=10, help="number of turns used as context")
    parser.add_argument('--batch_size', type=int, default=500, help="number of batch size")

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
    system_file_path = '../multiwoz/system/' + str(process_id) + '/'
    data_set = ''
    system = ''
    model = None

    config = json.load(open(system_file_path + 'config.json', 'r'))
    data_set = config['dataset']
    system = config['system']

    model_path = 'model/' + system + '/'
    for file in os.listdir(model_path):
        if file.startswith('final'):
            model_name = file
            break
    args.system_file_path = system_file_path
    args.model_path = 'model/' + system + '/' + model_name + '/'
    args.max_turn = int(system.split('_')[-1].split('-')[-1])
    test(args)


if __name__ == '__main__':
    main()
