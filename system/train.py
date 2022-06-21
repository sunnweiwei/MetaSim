import torch
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    set_seed,
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
import os
import json
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from accelerate import Accelerator


def train(args):
    accelerator = Accelerator(fp16=args.fp16, cpu=args.cpu, device_placement=True)
    set_seed(args.seed)

    # tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    if args.special_tokens:
        for line in open(args.special_tokens, 'r', encoding='utf-8'):
            line = line.strip('\n')
            tokenizer.add_tokens(line)
    tokenizer.eos_token = args.eos_token
    tokenizer.pad_token = tokenizer.eos_token
    accelerator.print('Tokenizer vocab_size is {}'.format(tokenizer.vocab_size))

    # Data loader
    class MyDataset(Dataset):
        def __init__(self):
            accelerator.print('Start load examples...')

            self.examples = []
            self.labels = []
            self.token_ids = []
            self.attention_masks = []

            system_token_id = tokenizer.convert_tokens_to_ids(['system'])
            user_token_id = tokenizer.convert_tokens_to_ids(['user'])
            induction_token_id = tokenizer.convert_tokens_to_ids(['=>'])

            examples = json.load(open(args.train_data_file))

            data_size = int(len(examples) / 100.0 * int(args.domain))
            remove_size = len(examples) - data_size
            restaurant_total = 0
            taxi_total = 0
            for example in examples:
                belief = example['belief']
                if 'restaurant' in belief:
                    restaurant_total += 1
                    continue
                if 'taxi' in belief:
                    taxi_total += 1
            accelerator.print(f"restaurant_total: {restaurant_total}")
            accelerator.print(f'taxi_total: {taxi_total}')

            if remove_size >= restaurant_total:
                remove_size -= restaurant_total
                restaurant_total = 0
            else:
                restaurant_total -= remove_size
                remove_size = 0
            if remove_size >= taxi_total:
                remove_size -= taxi_total
                taxi_total = 0
            else:
                taxi_total -= remove_size
                remove_size = 0


            response_pool = []
            belief_pool = []
            for i in examples:
                response = i['reply'] + ' ' + tokenizer.eos_token
                response_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response))
                response_pool.append(response_id)

                belief_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(i['belief']))
                belief_pool.append(belief_id)

            for example in examples:
                history = example['history']
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

                belief = example['belief']

                belief_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(belief))
                # belief_id = belief_id[:args.max_bs]
                response = example['reply'] + ' ' + tokenizer.eos_token
                response_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response))
                # response_id = response_id[:args.max_rs]

                # num_remain = args.max_length - len(belief_id) - len(response_id) - 1
                # context_ids = context_ids[-num_remain:]
                # token_ids_for_context = token_ids_for_context[-num_remain:]

                token_id = token_ids_for_context + system_token_id + system_token_id * len(
                    belief_id) + system_token_id * len(response_id)
                source = context_ids + induction_token_id + belief_id + response_id
                if args.with_LM:
                    target = source
                else:
                    target = [-1] * len(context_ids) + [-1] * len(induction_token_id) + belief_id + response_id

                if len(source) < args.max_length:
                    attention_mask = [0] * args.max_length
                    attention_mask[:len(source)] = [1] * len(source)
                    source += [0] * (args.max_length - len(source))
                    target += [-1] * (args.max_length - len(target))
                    token_id += [0] * (args.max_length - len(token_id))
                else:
                    attention_mask = [1] * args.max_length
                    source = source[-args.max_length:]
                    target = target[-args.max_length:]
                    token_id = token_id[-args.max_length:]


                if not len(source) == len(target) == len(token_id) == len(attention_mask):
                    import pdb
                    pdb.set_trace()

                if 'restaurant' in belief:
                    if restaurant_total == 0:
                        continue
                    else:
                        self.examples.append(source)
                        self.labels.append(target)
                        self.token_ids.append(token_id)
                        self.attention_masks.append(attention_mask)
                        restaurant_total -= 1
                else:
                    if 'taxi' in belief:
                        if taxi_total == 0:
                            continue
                        else:
                            self.examples.append(source)
                            self.labels.append(target)
                            self.token_ids.append(token_id)
                            self.attention_masks.append(attention_mask)
                            taxi_total -= 1
                    else:
                        if remove_size > 0:
                            remove_size -= 1
                            continue
                        else:
                            self.examples.append(source)
                            self.labels.append(target)
                            self.token_ids.append(token_id)
                            self.attention_masks.append(attention_mask)

            accelerator.print(f'Total examples is {len(self.examples)}')


        def __len__(self):
            return len(self.examples)

        def __getitem__(self, item):
            return torch.tensor(self.examples[item]), torch.tensor(self.token_ids[item]), torch.tensor(
                self.labels[item]), torch.tensor(self.attention_masks[item]),

    train_dataset = MyDataset()
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
    )

    # model
    if not args.model_path:
        model_config = GPT2Config.from_json_file(args.model_config)
        model = GPT2LMHeadModel(config=model_config)
    else:
        model = GPT2LMHeadModel.from_pretrained(args.model_path)
    if args.special_tokens:
        model.resize_token_embeddings(len(tokenizer))
    accelerator.print('Model config: ' + model.config.to_json_string())

    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    accelerator.print('Number of parameters: {}'.format(num_parameters))

    # model = GPT2LMHeadModel.from_pretrained('newest_model/data-100_context-15/epoch15_0_tr0.36675771297809034')
    # if args.special_tokens:
    #     model.resize_token_embeddings(len(tokenizer))
    # accelerator.print('Model config: ' + model.config.to_json_string())
    # num_parameters = 0
    # parameters = model.parameters()
    # for parameter in parameters:
    #     num_parameters += parameter.numel()
    # accelerator.print('Number of parameters: {}'.format(num_parameters))


    # opt, scheduler
    args.total_steps =  len(train_loader) // args.gradient_accumulation_steps * args.epochs
    accelerator.print(f"Total steps is {args.total_steps}")
    optimizer = AdamW(params=model.parameters(), lr=args.lr, eps=args.adam_epsilon)
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps
    )
    # scheduler.

    # train
    accelerator.print("***** Running training *****")
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
            inputs, tokens, labels, masks = batch
            outputs = model(inputs, labels=labels, token_type_ids=tokens, attention_mask=masks)
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
                tk.set_postfix(epoch=epoch + 1, global_step=global_step, loss=(tr_loss - logging_loss)/args.log_step)
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
    parser.add_argument("--train_data_file", default='data/train.soloist_filled_15.json', type=str,
                        help="The input training data file.")
    parser.add_argument("--output_dir", default='newest_model/', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters

    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    # model parameters
    parser.add_argument('--model_config', default='model/pretrained/config.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument("--model_path", default="model/pretrained/", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--max_turn", default=15, type=int, help="")
    parser.add_argument("--mc_loss_efficient", default=0.2, type=float, help="")
    parser.add_argument("--num_candidates", default=1, type=int, help="")
    parser.add_argument("--add_special_action_tokens", default='', type=str)
    parser.add_argument("--add_same_belief_response_prediction", type=bool, default=True, help="")
    parser.add_argument("--add_response_prediction", type=bool, default=True, help="")
    parser.add_argument("--add_belief_prediction", type=bool, default=True, help="")
    parser.add_argument('--with_LM', type=bool, default=True, help="")
    parser.add_argument("--max_length", default=512, type=int, help="")
    parser.add_argument("--max_bs", default=40, type=int, help="")
    parser.add_argument("--max_rs", default=40, type=int, help="")
    parser.add_argument("--domain", default='1', type=int, help="")

    # tokenizer parameters
    parser.add_argument('--tokenizer_path', default='pretrained/vocab.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--eos_token', default='<|endoftext|>', type=str, required=False, help='eos_token')
    parser.add_argument('--special_tokens', default='data/special_tokens.txt', type=str, required=False,
                        help='special_tokens')

    # train parameters
    parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument('--lr', default=5e-5, type=float, required=False, help='学习率')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
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
