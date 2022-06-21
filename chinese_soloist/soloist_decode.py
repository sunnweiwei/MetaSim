import copy

from transformers import (
    set_seed,
    GPT2Config,
    GPT2LMHeadModel,
)
import json
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tokenizations.bpe_tokenizer import get_encoder
from accelerate import Accelerator, DistributedType
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def decode(args):
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

    # Data loader
    class MyDataset(Dataset):
        def __init__(self):
            super(Dataset, self).__init__()
            self.examples = []
            accelerator.print('Loading examples...')
            for line in tqdm(open(args.test_data_file, 'r', encoding='utf-8')):
                line.strip('\n')
                dialog = json.loads(line)
                current_turns = []
                for turn in dialog['turns']:
                    if turn['speaker'] == 'Q':
                        uttrance = '用户：' + turn['turn']
                    else:
                        uttrance = '系统：' + turn['turn']
                    if turn['speaker'] == 'A' and current_turns:
                        example = {}
                        example['history'] = copy.copy(current_turns)
                        example['reply'] = uttrance
                        self.examples.append(example)
                    current_turns.append(uttrance)
            num_remain = len(self.examples) // 10
            self.examples = self.examples[: num_remain]
            accelerator.print('Load examples done.')
            accelerator.print('Total number of examples is %s' % num_remain)

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, item):
            example = self.examples[item]
            history = example['history']
            context = history[-args.max_turn:]
            context = ''.join(context)
            source = context + '=>'
            encode_ans = tokenizer.encode_plus(source, truncation=False)
            encode_ans['input_ids'] = encode_ans['input_ids'][-args.max_context_length:]
            encode_ans['token_type_ids'] = encode_ans['token_type_ids'][-args.max_context_length:]
            encode_ans['attention_mask'] = encode_ans['attention_mask'][-args.max_context_length:]
            return encode_ans

    def collate_fn(examples):
        if accelerator.distributed_type == DistributedType.TPU:
            return tokenizer.pad(examples, padding="max_length", max_length=args.max_length, return_tensors="pt")
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    test_dataset = MyDataset()
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn
    )

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

    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

    # decode
    corpus = []
    model_corpus = []
    accelerator.print('Starting decoding...')
    model.eval()
    tk = tqdm(test_dataloader, total=len(test_dataloader), desc='batch')
    for batch in tk:
        outputs = model.generate(
            **batch,
            max_length=args.max_length,
            do_sample=True,
            num_return_sequences=args.num_samples,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k,
            top_p=args.top_p
        )
        decode_ans = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for ans in decode_ans:
            try:
                ans = ans.split('=>')[1].split('系统：')[1]
            except:
                ans = ''
            ans = ans.replace(' ', '')
            model_corpus.append(ans)
    try:
        save_name = args.model_path.split('/')[1]
        json.dump(model_corpus, open(f"{args.output_dir}{save_name}.json", 'w'))
        # for example in test_dataset.examples:
        #     corpus.append(example['reply'].split('系统：')[1])
        # json.dump(corpus, open(f"{args.output_dir}gt.json", 'w'))
    except:
        json.dump(model_corpus, open(f"decode.json", 'w'))
    # corpus_ = [[tokenizer.tokenize(i)] for i in corpus]
    # hypothesis_ = [tokenizer.tokenize(i) for i in model_corpus]
    # accelerator.print('BLEU score:', corpus_bleu(corpus_, hypothesis_, smoothing_function=SmoothingFunction().method1))

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

    args = parser.parse_args()
    decode(args)



if __name__ == '__main__':
    main()
