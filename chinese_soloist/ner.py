import copy
import os
import sys

sys.path += ['./']
import json
from tqdm import tqdm
import spacy
from collections import defaultdict
import jieba


def do_ner(this_data):
    entity = {}
    nlp = spacy.load('zh_core_web_md')
    for session in tqdm(this_data):
        session = json.loads(session)
        text = []
        for turn in session['turns']:
            text.append(turn['turn'].replace('【', ' ').replace('】', ' '))
        text = ' '.join(text)
        for ne in nlp(text).ents:
            entity[str(ne)] = str(ne.label_)
    return [[k, v] for k, v in entity.items()]

def mp(func, data, processes=20, **kwargs):
    import multiprocessing
    pool = multiprocessing.Pool(processes=processes)
    length = len(data) // processes + 1
    results = []
    for ids in range(processes):
        collect = data[ids * length:(ids + 1) * length]
        results.append(pool.apply_async(func, args=(collect,), kwds=kwargs))
    pool.close()
    pool.join()
    result_collect = []
    for j, res in enumerate(results):
        result = res.get()
        result_collect.extend(result)
    return result_collect


def main():
    data = [line for line in open('data/train_set.txt', encoding='utf-8')] + \
           [line for line in open('data/test_set.txt', encoding='utf-8')]
    print(len(data))
    results = mp(do_ner, data, processes=50)
    all_entity = {}
    for k, v in results:
        all_entity[k] = v
    print(len(all_entity))
    with open('data/entity.txt', 'w', encoding='utf-8') as f:
        for k, v in tqdm(all_entity.items()):
            f.write(f'{k}\t{v}\n')


def is_ch(word):
    for ch in word:
        if not ('\u4e00' <= ch <= '\u9fef') and not ('\u3400' <= ch <= '\u4db5') \
                and not ('\u20000' <= ch <= '\u2a6d6') and not ('\u2a700' <= ch <= '\u2b734') \
                and not ('\u2b740' <= ch <= '\u2b81d') and not ('\u2b820' <= ch <= '\u2cea1') \
                and not ('\u2ceb0' <= ch <= '\u2ebe0') and not ('\u2f00' <= ch <= '\u2fd5') \
                and not ('\u2e80' <= ch <= '\u2ef3') and not ('\uf900' <= ch <= '\ufad9') \
                and not ('\u2f800' <= ch <= '\u2fa1d') and not ('\ue815' <= ch <= '\ue86f') \
                and not ('\ue400' <= ch <= '\ue5e8') and not ('\ue600' <= ch <= '\ue6cf') \
                and not ('\u31c0' <= ch <= '\u31e3') and not ('\u2ff0' <= ch <= '\u2ffb') \
                and not ('\u3105' <= ch <= '\u312f') and not ('\u31a0' <= ch <= '\u31ba'):
            return False
            break
    return True


def divide():
    data = [line[:-1] for line in open('dataset/jddc/entity.txt', encoding='utf-8')]
    print(len(data))
    entity = defaultdict(list)
    for line in data:
        e, l = line.split('\t')
        entity[l].append(e)
    new_entity = []
    for l, e in entity.items():
        if l in ['DATE', 'CARDINAL', 'TIME', 'MONEY', 'ORDINAL', 'QUANTITY', 'LANGUAGE', 'PERCENT']:
            continue
        e = [item for item in e if is_ch(item) and 1 < len(item)]
        print(l, len(e), e[:20])
        new_entity.extend([[item, l] for item in e])
    print(len(new_entity))
    with open('dataset/jddc/cleaned_entity.txt', 'w', encoding='utf-8') as f:
        for k, v in tqdm(new_entity):
            f.write(f'{k}\t{v}\n')


def ltp_ner():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    args = parser.parse_args()
    ids = int(args.id)
    processes = 4

    data = [line for line in open('dataset/jddc/train_set.txt', encoding='utf-8')] + \
           [line for line in open('dataset/jddc/test_set.txt', encoding='utf-8')]
    length = len(data) // processes + 1
    data = data[ids * length:(ids + 1) * length]

    from ltp import LTP
    ltp = LTP()
    entity = {}
    collect = []
    for session in tqdm(data):
        session = json.loads(session)
        text = []
        for turn in session['turns']:
            text.append(turn['turn'].replace('【', ' ').replace('】', ' '))
        # text = ' '.join(text)
        # collect.append(text)
        collect.extend(text)
        if len(collect) >= 512:
            seg, hidden = ltp.seg(collect)
            ner = ltp.ner(hidden)
            for one, one_seg in zip(ner, seg):
                for tag, start, end in one:
                    entity["".join(one_seg[start:end + 1])] = str(tag)
            collect = []
    seg, hidden = ltp.seg(collect)
    ner = ltp.ner(hidden)
    for one, one_seg in zip(ner, seg):
        for tag, start, end in one:
            entity["".join(one_seg[start:end + 1])] = str(tag)
    with open(f'dataset/jddc/ltp_entity{ids}.txt', 'w', encoding='utf-8') as f:
        for k, v in tqdm(entity.items()):
            f.write(f'{k}\t{v}\n')


def merge():
    data = [line[:-1] for line in open('dataset/jddc/ltp_entity0.txt', encoding='utf-8')] + \
           [line[:-1] for line in open('dataset/jddc/ltp_entity1.txt', encoding='utf-8')] + \
           [line[:-1] for line in open('dataset/jddc/ltp_entity2.txt', encoding='utf-8')] + \
           [line[:-1] for line in open('dataset/jddc/ltp_entity3.txt', encoding='utf-8')]
    all_entity = {}
    for line in data:
        k, v = line.split('\t')
        all_entity[k] = v
    print(len(all_entity))
    entity = defaultdict(list)
    for e, l in all_entity.items():
        entity[l].append(e)
    new_entity = []
    for l, e in entity.items():
        e = [item.replace(' ', '') for item in e]
        e = [item for item in e if is_ch(item) and 1 < len(item)]
        print(l, len(e), e[:20])
        new_entity.extend([[item, l] for item in e])
    print(len(new_entity))
    with open('dataset/jddc/cleaned_ltp_entity.txt', 'w', encoding='utf-8') as f:
        for k, v in tqdm(new_entity):
            f.write(f'{k}\t{v}\n')


def show():
    data = [line[:-1] for line in open('dataset/jddc/cleaned_entity.txt', encoding='utf-8')]
    print(len(data))
    data2 = [line[:-1] for line in open('dataset/jddc/cleaned_ltp_entity.txt', encoding='utf-8')]
    print(len(data2))
    entity_set = set()
    for line in data:
        e, l = line.split('\t')
        entity_set.add(e)
    overlaped = []
    for line in data2:
        e, l = line.split('\t')
        if e not in entity_set:
            overlaped.append(e)
    print(len(overlaped))
    print(overlaped[:100])
    with open('dataset/jddc/ltp_only.txt', 'w', encoding='utf-8') as f:
        for k in tqdm(overlaped):
            f.write(f'{k}\n')


def add_entity(this_data, entity_set=None):
    new_data = []
    for session in tqdm(this_data):
        session = json.loads(session)
        new_session = []
        for turn in session['turns']:
            text = turn['turn'].replace('【', ' ').replace('】', ' ')
            this_ents = []
            all_word = list(set([ne for ne in jieba.cut(text)]))
            for ne in all_word:
                ne = str(ne)
                if ne in entity_set:
                    this_ents.append([ne, entity_set[ne]])
            turn['ents'] = this_ents
            new_session.append(copy.deepcopy(turn))
        new_data.append(json.dumps(new_session, ensure_ascii=False))
    return new_data


def build():
    entity_set = {}
    for line in open('data/final_entity.txt', encoding='utf-8'):
        line = line[:-1]
        e, l = line.split('\t')
        entity_set[e] = l
    data = [line[:-1] for line in open('data/test_set.txt', encoding='utf-8')]
    results = mp(add_entity, data, processes=50, entity_set=entity_set)
    with open('data/test_set_jieba.txt', 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')


def filer_entity():
    from spacy.lang.zh.stop_words import STOP_WORDS
    stop_slot = ['数字', '京东', '商品', '下单', '亲亲', '姓名', '电话', '服务', '客服', '单', '小', '大', '好滴',
                 '简单', '等等', '修改', '订单', '金额', '日期', '时间', '评价', '地址', '订单', '日'] + list(STOP_WORDS)
    stop_slot = set(stop_slot)
    entity_set = {}

    for line in open('dataset/jddc/ali_all_cleaned_entity.txt', encoding='utf-8'):
        line = line[:-1]
        e, l = line.split('\t')
        if e not in stop_slot:
            entity_set[e] = l
    with open('dataset/jddc/final_entity.txt', 'w', encoding='utf-8') as f:
        for e, l in tqdm(entity_set.items()):
            f.write(f'{e}\t{l}\n')


# entity_set = defaultdict(list)
# for line in open('dataset/jddc/final_entity.txt', encoding='utf-8'):
#     line = line[:-1]
#     e, l = line.split('\t')
#     entity_set[l].append(e)
# for l, e in entity_set.items():
#     print(l, len(e), e[:10])
# filer_entity()

# data = [line for line in open('dataset/jddc/train_set_jieba.txt', encoding='utf-8')]
# slots = defaultdict(int)
# num_null = 0
# num_all = 0
# for session in tqdm(data):
#     session = json.loads(session)
#     for turn in session:
#         num_all += 1
#         if len(turn['ents']) == 0:
#             num_null += 1
#         # for ent in turn['ents']:
#         #     slots[ent[0]] += 1
# print(num_null, num_all - num_null, num_all)
# print(num_null / num_all, (num_all - num_null) / num_all, num_all)
# slots = [[k, v] for k, v in slots.items()]
# slots.sort(key=lambda x: x[1], reverse=True)
# print(slots[:1000])

# # show()
# merge()
# divide()
# ltp_ner()
# main()
build()
