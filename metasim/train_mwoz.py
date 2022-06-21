import sys

sys.path += ['./']
from transformers import AdamW, Adafactor, get_linear_schedule_with_warmup
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5TokenizerFast
from accelerate import Accelerator
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.evaluation import f1_score, eval_all
from utils.io import read_pkl, read_file, write_pkl, write_file
import torch
from tqdm import tqdm
import os
import json
import copy
import csv
import logging
import re
from collections import OrderedDict
from nltk import word_tokenize
import warnings

warnings.filterwarnings("ignore")

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


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


def parse_act(act):
    if act is None:
        return 'none'
    out = []
    for key in act:
        action = key[key.index('-') + 1:].lower()
        values = []
        for slot in act[key]:
            if slot[0] == 'none':
                continue
            slot[0] = slot[0].replace("'", '')
            values.append(f'{slot[0]}')
        out.append(f'{action} ' + ' , '.join(values))
    out = ' | '.join(out).lower()
    if len(out) == 0:
        return 'none'
    return out


def norm_text(text):
    text = re.sub("/", " / ", text)
    text = re.sub("\-", " \- ", text)
    text = re.sub("theres", "there's", text)
    text = re.sub("dont", "don't", text)
    text = re.sub("whats", "what's", text)
    text = text.lower()
    tokens = word_tokenize(text)
    text = ' '.join(tokens)
    return text


def delex_text(text, span_info):
    if span_info is None:
        return text
    text = text.split()
    for span in span_info:
        # slot = span[0][span[0].index('-') + 1:] + ' ' + span[1]
        slot = span[1]
        slot = f'[{slot}]'.lower()
        span[2] = span[2].lower()
        if text[min(span[3], len(text) - 1)].startswith(span[2]):
            for i in range(span[3], min(span[4] + 1, len(text))):
                text[i] = '~' + slot
        else:
            for i in range(min(span[3], len(text) - 1) - 3, len(text)):
                if ' '.join(text[i:i + span[4] - span[3] + 1]).startswith(span[2]):
                    for j in range(i, i + span[4] - span[3] + 1):
                        text[j] = '~' + slot
                    break

    new_text = ''
    for item in text:
        if item in new_text and item.startswith('~'):
            continue
        new_text += ' ' + item
    new_text = new_text.replace('~', ' ')
    new_text = ' '.join(new_text.split())
    return new_text


def get_slot(span_info):
    if span_info is None:
        return []
    slots = []
    for span in span_info:
        slots.append(norm_text(span[2]))
    return slots


def load_data(file):
    data = json.load(open(file))
    train_data = []
    for key in tqdm(data):
        goal = data[key]['goal']
        session = data[key]['log']
        dialogue_his = []
        delex_his = []
        actions = []
        goal = parse_goal(goal)
        goal = f'[ {goal} ]'
        speaker = 0
        for turn in session:
            if speaker % 2:
                role = 'system: '
            else:
                role = 'user: '
            actions.append(role + parse_act(turn.get('dialog_act', None)))
            text = norm_text(turn['text'])
            delexed = delex_text(text, turn.get('span_info', None))
            slots = get_slot(turn.get('span_info', None))
            text = role + text
            delexed = role + delexed
            if not speaker % 2:
                train_data.append([goal, '. '.join(dialogue_his), text,
                                   ' ', copy.deepcopy(actions), delexed, '. '.join(delex_his), slots])
                # print(train_data[-1])
                # input('>')
            dialogue_his.append(text)
            delex_his.append(delexed)
            speaker += 1
            # print(delexed)
            # input('>')

        actions.append('user: end of dialogue')
        train_data.append([goal, '. '.join(dialogue_his), 'user: [END]',
                           ' ', copy.deepcopy(actions), 'user: [END]', '. '.join(delex_his), []])

    return train_data


class MWOZData(Dataset):
    def __init__(self, dialog, context_len=128, response_len=128, goal_len=None,
                 prompt_len=None, tokenizer=None, collector=None):
        super(Dataset, self).__init__()
        self.goal = [item[0] for item in dialog]
        self.context = [item[1] for item in dialog]
        self.response = [item[2] for item in dialog]
        self.prompt = [item[3] for item in dialog]
        self.actions = [item[4] for item in dialog]
        self.template = [item[5] for item in dialog]
        self.delex_context = [item[6] for item in dialog]
        self.slots = [item[7] for item in dialog]

        if len(dialog[0]) > 8:
            self.memory = [item[8] for item in dialog]
        else:
            self.memory = [''] * len(dialog)

        self.tokenizer = tokenizer

        self.context_len = context_len
        self.response_len = response_len
        self.goal_len = response_len if goal_len is None else goal_len
        self.prompt_len = prompt_len if prompt_len is None else self.goal_len
        self.collector = collector

    def replace_system_actions(self, actions):
        assert len(actions) == len(self)
        for index in range(len(self)):
            if len(self.actions[index]) > 1:
                self.actions[index][-2] = actions[index]

    def replace_user_actions(self, actions):
        assert len(actions) == len(self)
        for index in range(len(self)):
            self.actions[index][-1] = actions[index]

    def replace_prompts(self, prompts):
        assert len(prompts) == len(self)
        for index in range(len(self)):
            self.prompt[index] = prompts[index]

    def replace_templates(self, template):
        assert len(template) == len(self)
        for index in range(len(self)):
            self.template[index] = template[index]

    def __getitem__(self, index, **kwargs):
        goal = self.goal[index] if 'goal' not in kwargs else kwargs['goal']
        context = self.context[index] if 'context' not in kwargs else kwargs['context']
        response = self.response[index] if 'response' not in kwargs else kwargs['response']
        prompt = self.prompt[index] if 'prompt' not in kwargs else kwargs['prompt']
        actions = self.actions[index] if 'actions' not in kwargs else kwargs['actions']
        template = self.template[index] if 'template' not in kwargs else kwargs['template']
        delex_context = self.delex_context[index] if 'delex_context' not in kwargs else kwargs['delex_context']
        memory = self.memory[index] if 'memory' not in kwargs else kwargs['memory']

        goal = self.tokenizer.encode(goal, truncation=True, max_length=self.goal_len)
        context = self.tokenizer.encode(context)
        response = self.tokenizer.encode(response, truncation=True, max_length=self.response_len)
        prompt = self.tokenizer.encode(prompt, truncation=True, max_length=self.prompt_len)
        actions = [self.tokenizer.encode(item, truncation=True, max_length=self.response_len) for item in actions]
        template = self.tokenizer.encode(template, truncation=True, max_length=self.response_len)
        delex_context = self.tokenizer.encode(delex_context)
        memory = self.tokenizer.encode(memory, truncation=True, max_length=self.prompt_len)

        inputs, targets = self.collector(goal, context, response,
                                         prompt, actions, template, delex_context, memory, self.context_len)
        inputs = inputs[:self.context_len]
        targets = targets[:self.response_len]
        inputs = torch.tensor(inputs)
        targets = torch.tensor(targets)

        return inputs, targets

    def __len__(self):
        return len(self.context)

    @staticmethod
    def collate_fn(data):
        context, response = zip(*data)
        context = pad_sequence(context, batch_first=True, padding_value=0)
        return {
            'input_ids': context,
            'attention_mask': context.ne(0),
            'labels': pad_sequence(response, batch_first=True, padding_value=-100),
        }


def nlu_collector(goal=None, context=None, response=None, prompt=None, actions=None, template=None,
                  delex_context=None, memory=None, context_len=512):
    inputs = goal + context[-(context_len - len(goal)):]
    if len(actions) < 2:
        targets = [358, 10, 5839, 1]
    else:
        targets = actions[-2]
    return inputs, targets


def policy_collector(goal=None, context=None, response=None, prompt=None, actions=None, template=None,
                     delex_context=None, memory=None, context_len=512):
    context_actions = sum(actions[:-1], [])[-(context_len - len(goal)):]
    context = context[-(context_len - len(goal) - len(context_actions) - len(template)):]
    inputs = goal + context + context_actions + template
    targets = actions[-1]
    return inputs, targets


def teacher_policy_collector(goal=None, context=None, response=None, prompt=None, actions=None, template=None,
                             delex_context=None, memory=None, context_len=512):
    context_actions = sum(actions[:-1], [])[-(context_len - len(goal) - len(prompt) - len(response)):]
    context = context[-(context_len - len(goal) - len(prompt) - len(context_actions) - len(response)):]
    inputs = goal + context + context_actions + prompt + response
    targets = actions[-1]
    return inputs, targets


def metaphor_collector(goal=None, context=None, response=None, prompt=None, actions=None, template=None,
                       delex_context=None, memory=None, context_len=512):
    inputs = context[-(context_len - len(actions[-1]) - len(memory)):] + \
             actions[-1] + memory
    targets = template
    return inputs, targets


def nlg_collector(goal=None, context=None, response=None, prompt=None, actions=None, template=None,
                  delex_context=None, memory=None, context_len=512):
    inputs = goal + context[-(context_len - len(goal) - len(actions[-1]) - len(memory)):] + memory + actions[-1]
    targets = response
    return inputs, targets


def prompt_nlg_collector(goal=None, context=None, response=None, prompt=None, actions=None, template=None,
                         delex_context=None, memory=None, context_len=512):
    inputs = goal + context[-(context_len - len(goal) - len(actions[-1]) - len(memory) - len(prompt)):] + memory + \
             actions[-1] + prompt
    targets = response
    return inputs, targets


def end2end_collector(goal=None, context=None, response=None, prompt=None, actions=None, template=None,
                      delex_context=None, memory=None, context_len=512):
    inputs = goal + context[-(context_len - len(goal)):]
    targets = response
    return inputs, targets


def no_preference_collector(goal=None, context=None, response=None, prompt=None, actions=None, template=None,
                            delex_context=None, memory=None, context_len=512):
    inputs = context[-context_len:]
    targets = response
    return inputs, targets


def no_policy_collector(goal=None, context=None, response=None, prompt=None, actions=None, template=None,
                        delex_context=None, memory=None, context_len=512):
    context_actions = sum(actions[:-1], [])[-(context_len - len(goal)):]
    context = context[-(context_len - len(goal) - len(context_actions)):]
    inputs = goal + context + context_actions
    targets = response
    return inputs, targets


def ablation_policy_collector(goal=None, context=None, response=None, prompt=None, actions=None, template=None,
                              delex_context=None, memory=None, context_len=512):
    context = context[-(context_len - len(goal) - len(prompt)):]
    inputs = goal + context + prompt
    targets = actions[-1]
    return inputs, targets


def ablation_nlg_collector(goal=None, context=None, response=None, prompt=None, actions=None, template=None,
                           delex_context=None, memory=None, context_len=512):
    if prompt is None:
        prompt = []
    inputs = goal + context[-(context_len - len(goal) - len(actions[-1]) - len(prompt)):] + actions[-1] + prompt
    targets = response
    return inputs, targets


def add_retrieval(data, file):
    if file is None:
        return data
    file = json.load(open(file))
    assert len(data) == len(file)
    out = []
    for line, ret in zip(data, file):
        ret = '. '.join(ret)
        line = line + [ret]
        out.append(line)
    return out


def train():
    accelerator = Accelerator()
    batch_size = 6
    epochs = 30
    save_path = 'ckpt/mwoz-nlg-final-v2'
    print(save_path)
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    # optimizer = AdamW(generator.parameters(), 1e-4)
    optimizer = Adafactor(model.parameters(), lr=1e-3, relative_step=False, scale_parameter=False)
    # optimizer = Adafactor(model.parameters())

    dataset = MWOZData(
        # load_data('dataset/mwoz/MultiWOZ_2.1/train_data.json'),
        add_retrieval(load_data('dataset/mwoz/MultiWOZ_2.1/train_data.json'),
                      'dataset/mwoz/train_template.json'),
        context_len=512, response_len=128, goal_len=128, prompt_len=192,
        tokenizer=tokenizer, collector=nlg_collector)

    # dataset.replace_templates([line[:-1] for line in open('ckpt/mwoz-metaphor-train/text/10.txt')])

    accelerator.print(f'data size={len(dataset)}')
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=batch_size,
                                              shuffle=True, num_workers=8)

    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=epochs * len(data_loader))
    # scheduler = None

    os.makedirs(save_path, exist_ok=True)

    accelerator.print(tokenizer.decode(dataset[128][0]))
    accelerator.print('==>')
    accelerator.print(tokenizer.decode(dataset[128][1]))

    for epoch in range(epochs):
        accelerator.print(f'Training epoch {epoch}')
        accelerator.wait_for_everyone()
        model.train()
        tk0 = tqdm(data_loader, total=len(data_loader))
        loss_report = []
        for batch in tk0:
            out = model(**batch)
            loss = out.loss
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_report.append(loss.item())
            tk0.set_postfix(loss=sum(loss_report) / len(loss_report))
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            accelerator.save(accelerator.unwrap_model(model).state_dict(), f'{save_path}/{epoch}.pt')


def lower(text):
    if isinstance(text, str):
        text = norm_text(text)
        text = text.replace('user:', '').replace('system:', '')
        text = text.replace('user :', '').replace('system :', '')
        return text
    return [lower(item) for item in text]


def strip(line):
    return [i.strip() for i in line]


def eval_action(predicts, answers):
    action_acc = []
    slot_acc = []
    for pred, ans in zip(predicts, answers):
        pred = {item.split()[0]: strip(' '.join(item.split()[1:]).split(',')) for item in pred.split(' | ')}
        ans = {item.split()[0]: strip(' '.join(item.split()[1:]).split(',')) for item in ans.split(' | ')}
        action_acc.append(int(all([(key in pred) for key in ans])))
        if action_acc[-1] == 1:
            slot_acc.append(int(all([all([(slot in pred[key]) for slot in ans[key]]) for key in ans])))
        else:
            slot_acc.append(0)
    return {'action_acc': sum(action_acc) / len(action_acc) * 100,
            'slot_acc': sum(slot_acc) / len(slot_acc) * 100}


def eval_slot_match(predicts, answers, skip_null=True):
    match = []
    for pred, ans in zip(predicts, answers):
        if len(ans) == 0 and skip_null:
            continue
        match.append(int(all([slot in pred for slot in ans])))
    return {'slot_match': sum(match) / len(match) * 100}


def test():
    batch_size = 32
    save_path = 'ckpt/mwoz-nlg-final-v2'
    # out_path = save_path + '-agenda'
    out_path = save_path
    tokenizer = T5TokenizerFast.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    dataset = MWOZData(add_retrieval(load_data('dataset/mwoz/MultiWOZ_2.1/test_data.json'),
                                     # None, ),
                                     'dataset/mwoz/test_template.json'),
                       context_len=512, response_len=256, goal_len=256, prompt_len=256,
                       tokenizer=tokenizer, collector=nlg_collector
                       )

    # dataset.replace_system_actions([line[:-1] for line in open('ckpt/mwoz-nlu-final/text/13.txt')])
    dataset.replace_user_actions([line[:-1] for line in open('ckpt/mwoz-policy-final/text/6.txt')])
    # dataset.replace_templates([line[:-1] for line in open('ckpt/mwoz-metaphor/text/10.txt')])

    # dataset.replace_user_actions([line[:-1] for line in open('ckpt/mwoz-agenda/text/0.txt')])
    # dataset.replace_templates([line[:-1] for line in open('ckpt/agenda-template/text/0.txt')])

    # dataset.replace_user_actions([line[:-1] for line in open('ckpt/mwoz-policy/text/10.txt')])
    # dataset.replace_templates([line[:-1] for line in open('ckpt/policy-template/text/0.txt')])

    model = model.cuda()

    for epoch in range(100, 0, -1):
        # if os.path.exists(f'{out_path}/text/{epoch}.txt'):
        #     print(f'eval model {save_path}/{epoch}.pt')
        #     output_predict = [line[:-1] for line in open(f'{out_path}/text/{epoch}.txt')]
        #     output_labels = [line[:-1] for line in open(f'{out_path}/text/true.txt')]
        #     print(eval_action(lower(output_predict), lower(output_labels)))
        #     print(eval_all(lower(output_predict), lower(dataset.response)))
        #     print(eval_slot_match(lower(output_predict), dataset.slots))
        # continue
        if os.path.exists(f'{save_path}/{epoch}.pt'):
            print(f'eval model {save_path}/{epoch}.pt, {out_path}')

            model.load_state_dict(
                OrderedDict({k: v for k, v in torch.load(f'{save_path}/{epoch}.pt').items()}))
            model.eval()
            data_loader = torch.utils.data.DataLoader(
                dataset,
                collate_fn=dataset.collate_fn,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8)
            tk0 = tqdm(data_loader, total=len(data_loader))
            output_predict = []
            output_labels = []
            with torch.no_grad():
                for batch in tk0:
                    predict = model.generate(
                        input_ids=batch['input_ids'].cuda().long(),
                        attention_mask=batch['attention_mask'].cuda(),
                        decoder_start_token_id=0,
                        # num_beams=3,
                        max_length=128)
                    predict = predict.cpu().tolist()
                    predict = tokenizer.batch_decode(predict, skip_special_tokens=True)
                    output_predict.extend(predict)

                    batch['labels'][batch['labels'] == -100] = 0
                    # output_labels.extend(tokenizer.batch_decode(batch['labels'], skip_special_tokens=True))
            # print(eval_action(lower(output_predict), lower(output_labels)))

            output_labels = dataset.response
            print(eval_all(lower(output_predict), lower(output_labels)))
            print(eval_slot_match(lower(output_predict), dataset.slots))

            write_file(output_predict, f'{out_path}/text/{epoch}.txt')
            write_file(output_labels, f'{out_path}/text/true.txt')


if __name__ == '__main__':
    # train()
    test()
#
