import json,copy
from collections import defaultdict, OrderedDict
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm


def parse_decoding_results(filename, mode):
    predictions = json.load(open(filename))
    # test_file_with_idx = [prediction.strip() for prediction in open('utils/multiwoz.test.idx.txt')]
    test_file_with_idx = json.load(open(f'{mode}.idx.json'))
    res = defaultdict(list)
    res_bs = defaultdict(list)
    for prediction, file_idx in tqdm(zip(predictions, test_file_with_idx), desc='parse_files'):
        filename = file_idx#.split('@@@@@@@@@')[0].strip().split()[0]
        candidates = []
        candidates_bs = []
        belief_state = {}
        for predict in prediction:
            predict = predict.strip()
            if 'system :' in predict:
                system_response = predict.split('system :')[-1]
            else:
                system_response = ''
            system_response = ' '.join(word_tokenize(system_response))
            system_response = system_response.replace('[ ','[').replace(' ]',']')
            candidates.append(system_response)
            
        for predict in prediction:
            predict = predict.strip().split('system :')[0]
            predict = ' '.join(predict.split()[:])
            predict = predict.strip().split('belief :')[1]
            domains = predict.split('|')
            belief_state = {}
            for domain_ in domains:
                if domain_ == '':
                    continue
                if len(domain_.split()) == 0:
                    continue
                domain = domain_.split()[0]
                if domain == 'none':
                    continue
                belief_state[domain] = {}
                svs = ' '.join(domain_.split()[1:]).split(';')
                for sv in svs:
                    if sv.strip() == '':
                        continue
                    sv = sv.split(' = ')
                    if len(sv) != 2:
                        continue
                    else:
                        s,v = sv
                    s = s.strip()
                    v = v.strip()
                    if v == "" or v == "dontcare" or v == 'not mentioned' or v == "don't care" or v == "dont care" or v == "do n't care" or v == 'none':
                        continue
                    if v.lower() == "night club":
                        v = "nightclub"
                    belief_state[domain][s] = v
            candidates_bs.append(copy.copy(belief_state))

        def compare(key1, key2):
            key1 = key1[1]
            key2 = key2[1]
            if key1.count('[') > key2.count('['):
                return 1
            elif key1.count('[') == key2.count('['):
                return 1 if len(key1.split()) > len(key2.split()) else -1
            else:
                return -1

        import functools
        candidates_w_idx = [(idx, v) for idx,v in enumerate(candidates)]
        candidates = sorted(candidates_w_idx, key=functools.cmp_to_key(compare))
        if len(candidates) != 0:
            idx, value = candidates[-1]
            candidates_bs = candidates_bs[idx]
            candidates = value
        
        filename = filename.split('.')[0]
        res[filename].append(candidates)
        res_bs[filename].append(candidates_bs)

    response = OrderedDict(sorted(res.items()))
    belief = OrderedDict(sorted(res_bs.items()))

    return response, belief


def parse_decoding_results_direct(predicts):
    candidates = []
    candidates_bs = []
    for predict in predicts:
        predict = predict.strip()
        if '系统:' in predict:
            system_response = predict.split('系统:')[-1]
        else:
            system_response = ''
        system_response = ' '.join(word_tokenize(system_response))
        candidates.append(system_response)

    def compare(key1, key2):
        key1 = key1[1]
        key2 = key2[1]
        return 1 if len(key1.split()) > len(key2.split()) else -1

    import functools
    candidates_w_idx = [(idx, v) for idx, v in enumerate(candidates)]
    candidates = sorted(candidates_w_idx, key=functools.cmp_to_key(compare))
    if len(candidates) != 0:
        idx, value = candidates[-1]
        candidates = value

    return candidates