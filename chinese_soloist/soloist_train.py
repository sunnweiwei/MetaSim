import copy

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    set_seed,
    GPT2Config,
    GPT2LMHeadModel,
)
import os
import json
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tokenizations.bpe_tokenizer import get_encoder
from accelerate import Accelerator, DistributedType

def train(args):
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
    accelerator.print('Tokenizer vocab_size is {}'.format(tokenizer.vocab_size))

    # Data loader
    class MyDataset(Dataset):
        def __init__(self):
            super(Dataset, self).__init__()
            self.examples = []
            accelerator.print('Loading examples...')
            for line in tqdm(open(args.train_data_file, 'r', encoding='utf-8')):
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
                        example['state'] = turn['state']
                        example['reply'] = uttrance
                        self.examples.append(example)
                    current_turns.append(uttrance)

            accelerator.print('Load examples done.')
            data_size = int(len(self.examples) / 100.0 * int(args.domain))
            self.examples = self.examples[:data_size]
            accelerator.print('Total number of examples is %s' % data_size)

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, item):
            example = self.examples[item]
            response = example['reply'] + ' ' + tokenizer.eos_token
            state = '状态：' + '；'.join(example['state'])
            history = example['history']
            context = history[-args.max_turn:]
            context = ''.join(context)
            source = context + '=>' + state + response
            encode_ans = tokenizer.encode_plus(
                source,
                truncation=False,
            )
            encode_ans['input_ids'] = encode_ans['input_ids'][-args.max_length:]
            encode_ans['token_type_ids'] = encode_ans['token_type_ids'][-args.max_length:]
            encode_ans['attention_mask'] = encode_ans['attention_mask'][-args.max_length:]
            return encode_ans

    def collate_fn(examples):
        if accelerator.distributed_type == DistributedType.TPU:
            return tokenizer.pad(examples, padding="max_length", max_length=args.max_length, return_tensors="pt")
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    train_dataset = MyDataset()
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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

    # opt, scheduler
    args.total_steps =  len(train_loader) // args.gradient_accumulation_steps * args.epochs
    accelerator.print(f"Total steps is {args.total_steps}")
    optimizer = AdamW(params=model.parameters(), lr=args.lr, correct_bias=args.correct_bias)
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps
    )

    # train
    accelerator.print('Starting training...')
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    for epoch in range(args.epochs):
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            model_loss = tr_loss / global_step if global_step > 0 else -1
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir + f'epoch{epoch + 1}_0_tr{model_loss}',
                                            save_function=accelerator.save,
                                            state_dict=accelerator.get_state_dict(model))
        model.train()
        tk = tqdm(enumerate(train_loader), total=len(train_loader), desc='batch')
        tk.set_postfix(epoch=epoch + 1, loss='')
        for step, batch in tk:
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            if global_step % args.log_step == 0:
                tk.set_postfix(epoch=epoch + 1, global_step=global_step, loss=(tr_loss - logging_loss) / args.log_step)
                logging_loss = tr_loss
            if global_step % args.save_step == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    model_loss = tr_loss / global_step if global_step > 0 else -1
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(args.output_dir + f'epoch{epoch + 1}_{global_step}_tr{model_loss}',
                                                    save_function=accelerator.save,
                                                    state_dict=accelerator.get_state_dict(model))
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        model_loss = tr_loss / global_step if global_step > 0 else -1
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir + f'final_tr{model_loss}',
                                        save_function=accelerator.save,
                                        state_dict=accelerator.get_state_dict(model))


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default='data/train_set.txt', type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default='model/', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    # model parameters
    parser.add_argument('--model_config', default='pretrained_model/config.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument("--model_path", default="pretrained_model/", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--max_turn", default=15, type=int, help="")
    parser.add_argument("--mc_loss_efficient", default=1, type=float, help="")
    parser.add_argument("--num_candidates", default=1, type=int, help="")
    parser.add_argument("--add_special_action_tokens", default='', type=str)
    parser.add_argument("--add_same_belief_response_prediction", action='store_true')
    parser.add_argument("--add_response_prediction", action='store_true')
    parser.add_argument("--add_belief_prediction", action='store_true')
    parser.add_argument('--with_LM', type=bool, default=True, help="")
    parser.add_argument("--max_length", default=256, type=int, help="")
    parser.add_argument("--domain", default='100', type=int, help="")

    # tokenizer parameters
    parser.add_argument('--tokenizer_path', default='pretrained_model/vocab.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--eos_token', default='<|endoftext|>', type=str, required=False, help='eos_token')
    parser.add_argument('--special_tokens', default='pretrained_model/special_tokens.txt', type=str, required=False,
                        help='special_tokens')
    parser.add_argument('--segment', default=False, action='store_true', help='中文以词为单位')
    parser.add_argument('--bpe_token', default=False, action='store_true', help='subword')
    parser.add_argument('--encoder_json', default="tokenizations/encoder.json", type=str, help="encoder.json")
    parser.add_argument('--vocab_bpe', default="tokenizations/vocab.bpe", type=str, help="vocab.bpe")

    # train parameters
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument("--batch_size", default=50, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--correct_bias', default=True, action='store_true', help='修正bias')
    parser.add_argument('--seed', default=42, type=int, required=False, help='随机数种子')
    parser.add_argument('--warmup_steps', default=0, type=int, required=False, help='warm up步数')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--log_step', default=500, type=int, required=False,
                        help='多少步汇报一次loss，设置为gradient accumulation的整数倍')
    parser.add_argument('--save_step', default=5000, type=int, required=False,
                        help='多少步保存一次模型，设置为gradient accumulation的整数倍')

    # accelerate parameters
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")

    args = parser.parse_args()
    args.output_dir += f'data-{args.domain}_context-{args.max_turn}/'
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)


if __name__ == '__main__':
    main()
