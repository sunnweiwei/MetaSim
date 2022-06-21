import sys

sys.path += ['./']
import torch
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, T5TokenizerFast
from collections import OrderedDict
from mwoz_driver.train_mwoz import MWOZData, load_data, nlg_collector, end2end_collector, parse_goal
from mwoz_driver.train_mwoz import no_preference_collector, no_policy_collector, pad_sequence
from mwoz_driver.train_mwoz import nlu_collector, policy_collector, ablation_nlg_collector
from mwoz_driver.test_agenda import AgendaPolicy
from mwoz_driver.retrieval import RetModel
from mwoz_driver.metaphor import ActionSearcher
from mwoz_driver.train_rerank import TestRankData, rerank_collector
import torch.nn.functional as F
import numpy as np
import json
import datetime
import copy
import transformers
import requests
import random
import time


def load_goal_data(file):
    goals = json.load(open(file))
    train_data = []
    for goal in goals:
        goal = parse_goal(goal)
        goal = f'[ {goal} ]'
        train_data.append([goal, '', '', '', '', '', '', []])
    return train_data


def load_raw_data(file):
    data = json.load(open(file))
    train_data = []
    for key in data:
        goal = data[key]['goal']
        goal = parse_goal(goal)
        goal = f'[ {goal} ]'
        train_data.append([goal, '', '', '', '', '', '', []])
    return train_data


def write_file(data, file):
    with open(file, 'w+') as f:
        for line in data:
            f.write(line + '\n')


class MyData(MWOZData):
    def process(self, index=0, **kwargs):
        return super().__getitem__(index, **kwargs)

    def get_goal(self, index):
        return self.goal[index]

    def __len__(self):
        return len(self.goal)


class OurPipe:
    def __init__(self, data_list, model_list, tokenizer, model_type=None):
        #  nlu policy nlg
        self.data_list = data_list
        self.model_list = model_list
        for model in self.model_list:
            try:
                model.eval()
            except:
                pass
        self.tokenizer = tokenizer
        self.model_type = model_type if model_type is not None else ['t5', 't5', 't5']

    def generate(self, index, context, actions):
        # nlu
        if self.model_type[0] == 't5':
            inputs, _ = self.data_list[0].process(index=index, context=context)
            inputs = inputs.long().cuda().unsqueeze(0)
            predict = self.model_list[0].generate(input_ids=inputs, max_length=128)
            predict = predict.cpu()
            system_action = self.tokenizer.batch_decode(predict, skip_special_tokens=True)[0]
        else:
            system_action = self.model_list[0].generate(context=context)

        if self.model_type[1] == 'skip':
            user_action = 'user: none'
        elif self.model_type[1] == 't5':
            _actions = actions + [system_action] + ['none']
            inputs, _ = self.data_list[1].process(index=index, context=context, actions=_actions)
            inputs = inputs.long().cuda().unsqueeze(0)
            predict = self.model_list[1].generate(input_ids=inputs, max_length=128)
            predict = predict.cpu()
            user_action = self.tokenizer.batch_decode(predict, skip_special_tokens=True)[0]
        else:
            _actions = actions + [system_action] + ['none']
            user_action = self.model_list[1].generate(_actions, self.data_list[1].goal[index])

        if self.model_type[2] == 't5':
            _actions = actions + [system_action, user_action]
            inputs, _ = self.data_list[2].process(index=index, context=context, actions=_actions)
            inputs = inputs.long().cuda().unsqueeze(0)
            predict = self.model_list[2].generate(
                input_ids=inputs, max_length=128, min_length=5, no_repeat_ngram_size=5)
            predict = predict.cpu()
            user_sentence = self.tokenizer.batch_decode(predict, skip_special_tokens=True)[0]
        else:
            _actions = actions + [system_action, user_action]
            user_sentence = self.model_list[2].generate(_actions, context, self.data_list[2].goal[index], fill=True)

        actions = actions + [system_action, user_action]

        return user_sentence, actions


class MetaPipe:
    def __init__(self, data_list, model_list, tokenizer, model_type=None, reranker=None, reranker_dataset=None):
        #  nlu policy nlg
        self.data_list = data_list
        self.model_list = model_list
        for model in self.model_list:
            try:
                model.eval()
            except:
                pass
        self.tokenizer = tokenizer
        self.model_type = model_type if model_type is not None else ['t5', 't5', 't5']
        dataset = MWOZData(load_data('dataset/mwoz/MultiWOZ_2.1/train_data.json'), )
        corpus = []
        for actions, context, template in zip(dataset.actions, dataset.delex_context, dataset.template):
            action = actions[-1]
            corpus.append([[action, context], template])
        self.searcher = ActionSearcher(corpus)
        self.reranker = reranker
        self.reranker_dataset = reranker_dataset

    def generate(self, index, context, actions):
        inputs, _ = self.data_list[0].process(index=index, context=context)
        inputs = inputs.long().cuda().unsqueeze(0)
        predict = self.model_list[0].generate(input_ids=inputs, max_length=128)
        predict = predict.cpu()
        system_action = self.tokenizer.batch_decode(predict, skip_special_tokens=True)[0]
        _actions = actions + [system_action] + ['none']
        metaphor = self.searcher.search(_actions, context)
        if self.reranker is not None:
            _actions = actions + [system_action] + ['none']
            inputs, targets = self.reranker_dataset.__getitem__(index=index, context=context, actions=_actions,
                                                                memory=metaphor, template='')
            inputs = inputs[1:]
            targets = targets[1:]
            inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
            batch = {
                'input_ids': inputs,
                'attention_mask': inputs.ne(0),
                'labels': pad_sequence(targets, batch_first=True, padding_value=-100),
            }
            num_psg = len(inputs)
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                out = self.reranker(**batch)
            logits = out.logits[:, 0, :]
            logits = logits.view(num_psg, 32128)
            logits = F.softmax(logits, dim=-1)
            yes_prob = logits[:, 4273]
            no_prob = logits[:, 150]
            prob = yes_prob
            prob = [-p for p in prob.cpu().tolist()]
            metaphor = [metaphor[ii] for ii in np.argsort(prob)]
            # print(prob)

        metaphor = '. '.join(metaphor)

        _actions = actions + [system_action] + ['none']
        inputs, _ = self.data_list[1].process(index=index, context=context, actions=_actions, memory=metaphor)
        inputs = inputs.long().cuda().unsqueeze(0)
        predict = self.model_list[1].generate(input_ids=inputs, max_length=128)
        predict = predict.cpu()
        user_action = self.tokenizer.batch_decode(predict, skip_special_tokens=True)[0]

        _actions = actions + [system_action, user_action]
        inputs, _ = self.data_list[2].process(index=index, context=context, actions=_actions, memory=metaphor)
        inputs = inputs.long().cuda().unsqueeze(0)
        predict = self.model_list[2].generate(
            input_ids=inputs, max_length=128, min_length=5, no_repeat_ngram_size=5)
        predict = predict.cpu()
        user_sentence = self.tokenizer.batch_decode(predict, skip_special_tokens=True)[0]

        actions = actions + [system_action, user_action]

        return user_sentence, actions


class End2endPipe:
    def __init__(self, data_list, model_list, tokenizer, model_type=None):
        #  nlu policy nlg
        self.data_list = data_list
        self.model_list = model_list
        for model in self.model_list:
            try:
                model.eval()
            except:
                pass

        self.tokenizer = tokenizer
        self.model_type = model_type if model_type is not None else ['t5']

    def generate(self, index, context, actions):
        inputs, _ = self.data_list[0].process(index=index, context=context)
        inputs = inputs.long().cuda().unsqueeze(0)
        predict = self.model_list[0].generate(
            input_ids=inputs, max_length=128, min_length=5, no_repeat_ngram_size=5, )
        predict = predict.cpu()
        user_sentence = self.tokenizer.batch_decode(predict, skip_special_tokens=True)[0]
        return user_sentence, actions


def call_api(context, goal=None, cache=None, config=None, local=False):
    if len(context.strip()) == 0:
        return '', '{}'
    if local:
        output = input('>> ')
        cache = '{}'
        return output, cache

    while True:
        try:
            url = "http://XXXXX"
            if len(cache) == 0:
                cache = '{}'
            data = {"context": context, 'goal': goal, 'cache': cache, **config}
            res = requests.post(url=url, data=data)
            res = json.loads(res.text)
            output = res['text']
            break
        except:
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'error')
            time.sleep(1)
            pass
    cache = res['cache']
    return output, cache


def post_goal(goal, context, index, cache, config):
    try:
        url = "http://XXXXX"
        if len(cache) == 0:
            cache = '{}'
        data = {"context": context, "goal": json.dumps(goal), 'index': index, 'cache': cache, **config}
        res = requests.post(url=url, data=data)
    except:
        pass


def post_print(*content):
    content = ' '.join(content)
    print(content)


def build_model(name, tokenizer, data, data_config):
    if name == 'end2end':
        dataset = MyData(data, **data_config, collector=end2end_collector)
        end2end_model = T5ForConditionalGeneration.from_pretrained('t5-base').cuda()
        end2end_model.load_state_dict(torch.load(f'ckpt/mwoz-end2end-new/19.pt'))
        pipeline = End2endPipe(data_list=[dataset], model_list=[end2end_model], tokenizer=tokenizer, model_type=['t5'])
        return pipeline, dataset
    elif name == 'preference':
        dataset = MyData(data, **data_config, collector=no_preference_collector)
        end2end_model = T5ForConditionalGeneration.from_pretrained('t5-base').cuda()
        end2end_model.load_state_dict(torch.load(f'ckpt/mwoz-no-preference/19.pt'))
        pipeline = End2endPipe(data_list=[dataset], model_list=[end2end_model], tokenizer=tokenizer, model_type=['t5'])
        return pipeline, dataset
    elif name == 'policy':
        nlu_dataset = MyData(data, **data_config, collector=nlu_collector)
        nlg_dataset = MyData(data, **data_config, collector=no_policy_collector)
        dataset = nlu_dataset
        nlu_model = T5ForConditionalGeneration.from_pretrained('t5-base').cuda()
        nlu_model.load_state_dict(torch.load(f'ckpt/mwoz-nlu-final/13.pt'))
        nlg_model = T5ForConditionalGeneration.from_pretrained('t5-base').cuda()
        nlg_model.load_state_dict(torch.load(f'ckpt/mwoz-no-policy/19.pt'))
        pipeline = OurPipe(data_list=[nlu_dataset, None, nlg_dataset],
                           model_list=[nlu_model, None, nlg_model], tokenizer=tokenizer,
                           model_type=['t5', 'skip', 't5'])
        return pipeline, dataset
    elif name == 'metaphor':
        nlu_dataset = MyData(data, **data_config, collector=nlu_collector)
        policy_dataset = MyData(data, **data_config, collector=policy_collector)
        nlg_dataset = MyData(data, **data_config, collector=nlg_collector)
        dataset = nlu_dataset

        nlu_model = T5ForConditionalGeneration.from_pretrained('t5-base').cuda()
        nlu_model.load_state_dict(torch.load(f'ckpt/mwoz-nlu-final/13.pt'))
        policy_model = T5ForConditionalGeneration.from_pretrained('t5-base').cuda()
        policy_model.load_state_dict(torch.load(f'ckpt/mwoz-policy-final/6.pt'))

        nlg_model = T5ForConditionalGeneration.from_pretrained('t5-base').cuda()
        nlg_model.load_state_dict(torch.load(f'ckpt/mwoz-nlg-final-v2/26.pt'))

        reranker_dataset = TestRankData(data, **data_config, neg_num=17, collector=rerank_collector)
        reranker = T5ForConditionalGeneration.from_pretrained('t5-base').cuda()
        reranker.load_state_dict(torch.load(f'ckpt/mwoz-rerank/29.pt'))
        # reranker = None

        pipeline = MetaPipe(data_list=[nlu_dataset, policy_dataset, nlg_dataset],
                            model_list=[nlu_model, policy_model, nlg_model], tokenizer=tokenizer,
                            model_type=['t5', 't5', 't5'], reranker=reranker, reranker_dataset=reranker_dataset)
        return pipeline, dataset
    else:
        nlu_dataset = MyData(data, **data_config, collector=nlu_collector)
        policy_dataset = MyData(data, **data_config, collector=policy_collector)
        nlg_dataset = MyData(data, **data_config, collector=ablation_nlg_collector)
        dataset = nlu_dataset

        nlu_model = T5ForConditionalGeneration.from_pretrained('t5-base').cuda()
        nlu_model.load_state_dict(torch.load(f'ckpt/mwoz-nlu-final/13.pt'))

        if name[1] == 't5':
            policy_model = T5ForConditionalGeneration.from_pretrained('t5-base').cuda()
            policy_model.load_state_dict(torch.load(f'ckpt/mwoz-policy-final/6.pt'))
        else:
            policy_model = AgendaPolicy()

        if name[2] == 't5':
            nlg_model = T5ForConditionalGeneration.from_pretrained('t5-base').cuda()
            nlg_model.load_state_dict(torch.load(f'ckpt/mwoz-ab-nlg-final/18.pt'))
        else:
            nlg_model = RetModel()

        pipeline = OurPipe(data_list=[nlu_dataset, policy_dataset, nlg_dataset],
                           model_list=[nlu_model, policy_model, nlg_model], tokenizer=tokenizer,
                           model_type=name)
        return pipeline, dataset


def test():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    args = parser.parse_args()
    this_id = int(args.id)
    total = 12

    task_name = 'end2end=10%'
    system_name = 'mwoz-domain-3'

    config = {'system': system_name, 'task': task_name}

    print(this_id, total, config)
    tokenizer = T5TokenizerFast.from_pretrained('t5-base')
    data = load_goal_data('dataset/mwoz/10w.json')
    data_config = dict(context_len=512, response_len=128, goal_len=128, prompt_len=128, tokenizer=tokenizer)
    user_key = 'user'
    system_key = 'system'
    sep_key = '. '

    pipeline, dataset = build_model(['t5', 't5', 't5'], tokenizer, data, data_config)

    dialogue_his = []
    actions = []
    index = this_id % total
    # index = random.randint(0, len(dataset) - 1)
    post_print('GOAL >>')
    cache = '{}'
    context = '. '.join([item for item in dialogue_his if len(item.strip()) != 0])
    context, _ = dataset.process(context=context, index=index, actions=['none'])
    post_print(str(tokenizer.batch_decode(context.long().unsqueeze(0), skip_special_tokens=True)))
    while True:
        system_res, cache = call_api(' [SEP] '.join(dialogue_his), dataset.get_goal(index), cache=cache, config=config)
        system_key = 'system'
        if not system_res.startswith(f'{system_key}: ') and len(system_res) > 0:
            system_res = f'{system_key}: ' + system_res
        system_res = system_res.replace(' -s', 's').replace(' -ly', 'ly')
        post_print('>>', system_res)
        if len(system_res) > 0:
            dialogue_his.append(system_res)
        context = sep_key.join([item.lower() for item in dialogue_his if len(item.strip()) != 0])

        sentence, actions = pipeline.generate(index=index, context=context, actions=actions)
        # sentence, actions = '', actions

        post_print('>>', sentence)
        dialogue_his.append(sentence)
        if len(dialogue_his) > 10 and 'all i need' in ''.join(dialogue_his).lower():
            sentence = f'{user_key}: [END]'
            post_print('>>', sentence)
            dialogue_his.append(sentence)
        if len(dialogue_his) > 100:
            sentence = f'{user_key}: [END]'
            post_print('>>', sentence)
            dialogue_his.append(sentence)
        if 'END' in sentence:
            # collect_data.append(copy.deepcopy(dialogue_his))
            # write_file(dialogue_his, 'dataset/dialog2.txt')
            post_goal(dataset.get_goal(index), ' [SEP] '.join(dialogue_his), index, cache=cache, config=config)
            post_print()
            index = index + total
            if index >= len(dataset):
                break
            index = index % len(dataset)
            # index = random.randint(0, len(dataset) - 1)
            dialogue_his = []
            actions = []
            post_print('GOAL >>')
            cache = '{}'
            context = '. '.join([item for item in dialogue_his if len(item.strip()) != 0])
            context, _ = dataset.process(context=context, index=index, actions=['none'])
            post_print(str(tokenizer.batch_decode(context.long().unsqueeze(0), skip_special_tokens=True)))

if __name__ == '__main__':
    test()
