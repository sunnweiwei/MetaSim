import sys

sys.path += ['./']
from mwoz_driver.train_mwoz import load_data, MWOZData
from gensim.summarization import bm25
from utils.evaluation import f1_score
from utils.mp import mp
import json
from tqdm import tqdm
from utils.io import write_file
import numpy as np


class ActionSearcher:
    def __init__(self, corpus):
        self.corpus = corpus
        self.bm25 = bm25.BM25([item[0][1].split()[-32:] for item in corpus])

    def _batch_search(self, data, **kwargs):
        out = []
        for line in tqdm(data):
            action, context, skip_idx = line
            out.append(self.search(action, context, skip_idx=skip_idx, **kwargs))
        return out

    def batch_search(self, action, context, skip_id_list=None, processes=20, **kwargs):
        if skip_id_list is None:
            skip_id_list = [None] * len(action)
        data = [[x, y, z] for x, y, z in zip(action, context, skip_id_list)]
        return mp(self._batch_search, data, processes=processes, **kwargs)

    def search(self, action, context, return_id=False, num=16, skip_idx=None):
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


def main():
    dataset = MWOZData(load_data('dataset/mwoz/MultiWOZ_2.1/train_data.json'), )
    corpus = []
    for actions, context, template in zip(dataset.actions, dataset.delex_context, dataset.template):
        action = actions[-2]
        corpus.append([[action, context], template])
    searcher = ActionSearcher(corpus)
    train = MWOZData(load_data('dataset/mwoz/MultiWOZ_2.1/train_data.json'), )

    skip_id_list = [i for i in range(len(actions))]
    response = [line[:-1] for line in open('cases/ours.txt')]
    metaphor_output = searcher.batch_search(actions, response, skip_id_list=None, num=16, processes=20)
    metaphor_output = [item for item in metaphor_output]

    json.dump(metaphor_output, open('dataset/mwoz/train_template.json', 'w'))


if __name__ == '__main__':
    main()
