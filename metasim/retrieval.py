import os
import sys

sys.path += ['./']
from mwoz_driver.train_mwoz import MWOZData, load_data, strip, lower, eval_action, write_file, eval_all, eval_slot_match
from collections import defaultdict
import numpy as np
from gensim.summarization import bm25
from sklearn import tree, ensemble
from tqdm import tqdm
from utils.mp import mp
from sklearn import metrics


class ActionSearcher:
    def __init__(self, corpus):
        self.corpus = corpus
        self.bm25 = bm25.BM25([item[0][1].split()[-32:] for item in corpus])

    def _batch_search(self, data, **kwargs):
        out = []
        for line in tqdm(data):
            action, context = line
            out.append(self.search(action, context, **kwargs))
        return out

    def batch_search(self, action, context, processes=20, **kwargs):
        data = [[x, y] for x, y in zip(action, context)]
        return mp(self._batch_search, data, processes=processes, **kwargs)

    def search(self, action, context, return_id=False, num=16, skip_idx=None):
        action = action[-2]
        action = self.norm_key(action)
        context = ' '.join(context.split()[-32:])
        context = self.norm_key(context)
        best_value = []
        best_idx = []
        best_score = 0
        for idx, (key, value) in enumerate(self.corpus):
            if skip_idx is not None and idx == skip_idx:
                continue
            key = key[0]
            score = self.score(action, key)
            if score == best_score:
                best_value.append(value)
                best_idx.append(idx)
            if score > best_score:
                best_score = score
                best_value = [value]
                best_idx = [idx]
        context = context.split()
        if len(best_value) > 0 and len(context) != 0:
            scores = [-self.bm25.get_score(context, idx) for idx in best_idx]
            best_value = [best_value[mi] for mi in np.argsort(scores)]
            best_idx = [best_idx[mi] for mi in np.argsort(scores)]
        best_value = best_value[:num]
        best_idx = best_idx[:num]
        if return_id:
            return best_value, best_idx
        return best_value

    @staticmethod
    def norm_key(key):
        key = key.replace('user:', '').replace('system:', '')
        return key

    @staticmethod
    def score(x, y):
        x = x.split()
        y = y.split()
        return sum([int(i in y) for i in x]) / max(len(x), 1)


class RetModel:
    def __init__(self):
        dataset = MWOZData(load_data('dataset/mwoz/MultiWOZ_2.1/train_data.json'), )
        corpus = []
        for actions, context, template in zip(dataset.actions, dataset.delex_context, dataset.template):
            action = actions[-1]
            corpus.append([[action, context], template])
        searcher = ActionSearcher(corpus)
        self.searcher = searcher

    def generate(self, test_actions, dataset_context, goal=None, fill=True):
        slot_map = {'price': 'pricerange', 'depart': 'departure', 'dest': 'destination',
                    'leave': 'leaveat', 'arrive': 'arriveby'}
        this_actions = [test_actions[-1]]
        dataset_context = [dataset_context]
        outputs = []
        for act, context in zip(this_actions, dataset_context):
            template = self.searcher.search(act, context)
            outputs.append(template)
        results = outputs[0][0]

        if fill:
            goal = goal.replace('[', '').replace(']', '').replace('|', ';')
            goal = [[y.strip().lower() for y in x.split('=')] for x in goal.split(';')]
            goal = [x for x in goal if len(x) > 1]
            clean_goal = [x for x in goal if x[1] not in dataset_context[0]]
            clean_goal = {x[0]: x[1] for x in clean_goal}
            all_goal = {x[0]: x[1] for x in goal}
            results = results.split()
            filled_results = []
            for item in results:
                if item[0] == '[' and item[-1] == ']':
                    item = item[1:-1]
                    item = slot_map.get(item, item)
                    if item in clean_goal:
                        item = clean_goal.get(item, item)
                    else:
                        item = all_goal.get(item, item)
                filled_results.append(item)
            results = ' '.join(filled_results)
        return results


def test():
    searcher = RetModel()

    test_actions = [[line[:-1]] for line in open('ckpt/mwoz-policy-final/text/6.txt')]
    # test_actions = [[line[:-1]] for line in open('ckpt/agenda/text/0.txt')]
    # test_actions = [line[:-1] for line in open('ckpt/mwoz-policy/text/10.txt')]

    dataset = MWOZData(load_data('dataset/mwoz/MultiWOZ_2.1/test_data.json'), )
    outputs = []
    for act, context, goal in tqdm(zip(test_actions, dataset.context, dataset.goal), total=len(test_actions)):
        template = searcher.generate(act, context, goal=goal, fill=True)
        outputs.append(template)

    print(eval_all(lower(outputs), lower(dataset.response)))
    print(eval_slot_match(lower(outputs), dataset.slots))

    os.makedirs(f'ckpt/mwoz-policy-final-ret/text', exist_ok=True)
    write_file(outputs, f'ckpt/mwoz-policy-final-ret/text/6.txt')


if __name__ == '__main__':
    # model = RetModel()
    # model.generate('user: none', 'system: hello',
    #                '[ domain = taxi ; destination = camboats ; departure = bangkok city ; arriveby = 13:30 | domain = attraction ; name = scudamores punting co | domain = restaurant ; food = british ; pricerange = moderate ; area = centre ]')
    test()
