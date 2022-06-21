import copy
import logging
import random
import sys

from evaluate import MultiWozDB
from nltk.tokenize import word_tokenize
from parse import parse_decoding_results_direct
from collections import defaultdict

db = MultiWozDB()
domain_attr = {
    'attraction': ['id', 'address', 'area', 'entrance fee', 'name', 'phone', 'postcode', 'pricerange', 'openhours', 'type'],
    'restaurant': ['id', 'address', 'area', 'food', 'introduction', 'name', 'phone', 'postcode', 'pricerange', 'location', 'type'],
    'hotel': ['id', 'address', 'area', 'internet', 'parking', 'single', 'double', 'family', 'name', 'phone', 'postcode', 'pricerange', 'takesbookings', 'stars', 'type'],
    'train': ['id', 'arriveBy', 'day', 'departure', 'destination', 'duration', 'leaveAt', 'price'],
}
domain_attr_all = {
    'attraction': ['id', 'address', 'area', 'entrance fee', 'name', 'phone', 'postcode', 'pricerange', 'openhours', 'type'],
    'restaurant': ['id', 'address', 'area', 'food', 'introduction', 'name', 'phone', 'postcode', 'pricerange', 'location', 'type'],
    'hotel': ['id', 'address', 'area', 'internet', 'parking', 'single', 'double', 'family', 'name', 'phone', 'postcode', 'pricerange', 'takesbookings', 'stars', 'type'],
    'train': ['id', 'arriveBy', 'day', 'departure', 'destination', 'duration', 'leaveAt', 'price'],
    'police': ['name', 'address', 'id', 'phone', 'postcode'],
    'taxi': ['type', 'phone'],
    'hospital': ['department', 'id', 'phone'],
}
police = {
    "name": "Parkside Police Station",
    "address": "Parkside, Cambridge",
    "id": 0,
    "phone": "01223358966",
    "postcode": "cb12dp"
}
taxi = [("toyota", "000000000", "time", "11:30")]
taxi_dict = {
    "type": "toyota",
    "phone": "000000000",
    "time": "11:30",
}
reference = '00000000'
value_replace = {
    'time': {
        'train': ['leaveAt', 'arriveBy']
    },
    'place': {
        'attraction': 'address',
        'restaurant': 'address',
        'hotel': 'address',
        'train': ['departure', 'destination'],
        'police': 'address'
    }
}
current_domain = ['taxi', 'train', 'hotel', 'restaurant', 'attraction', 'hospital', 'police']
eval_domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']

def fill(response, belief_state, talk):
    try:
        text = copy.deepcopy(response)
        slot_before = defaultdict(list)
        if talk is None:
            talk = 'hospital'
        response = response.split()
        for idx in range(len(response)):
            if response[idx].startswith('[') and response[idx].endswith(']'):
                slot = response[idx][1:-1]
                domain = slot.split('_')[0]
                attr = slot.split('_')[-1]
                if domain not in belief_state.keys():
                    if domain in domain_attr.keys() or domain == 'taxi':
                        domain = talk
                if domain in domain_attr_all.keys():
                    filled = get_value(domain, belief_state, attr, slot_before)
                    if filled:
                        response[idx] = str(filled)
                        if filled not in slot_before[domain + '_' + attr]:
                            slot_before[domain + '_' + attr].append(filled)
                        continue
                else:
                    if attr == 'count':
                        if talk in ['police', 'taxi']:
                            response[idx] = str(1)
                            continue
                        else:
                            number = str(len(db.queryResultVenues(talk, bs=belief_state[talk])))
                            if number == '0' and ('sorry' not in text and 'no' not in text):
                                response[idx] = str(1)
                            elif ("tickets" in text or "ticket" in text) and ('book' in text or 'booked' in text):
                                response[idx] = str(1)
                            else:
                                if slot_before[attr]:
                                    if number not in slot_before[attr]:
                                        response[idx] = str(number)
                                    else:
                                        number = int(slot_before[attr][-1]) // 2
                                        response[idx] = str(number if number > 0 else 1)
                                else:
                                    response[idx] = str(number)
                                slot_before[attr].append(number)
                            continue
                    else:
                        history = [response[i] for i in range(idx)]
                        history = ' '.join(history)
                        if attr == 'time':
                            if 'leave' in history:
                                filled = get_value('train', belief_state, 'leaveAt', slot_before)
                                if filled:
                                    response[idx] = str(filled)
                                    continue
                                filled = get_value('train', belief_state, 'leaveat', slot_before)
                                if filled:
                                    response[idx] = str(filled)
                                    continue
                            if 'arrive' in history:
                                filled = get_value('train', belief_state, 'arriveBy', slot_before)
                                if filled:
                                    response[idx] = str(filled)
                                    continue
                                filled = get_value('train', belief_state, 'arriveby', slot_before)
                                if filled:
                                    response[idx] = str(filled)
                                    continue
                        if 'booking' in belief_state.keys():
                            if attr in belief_state['booking'].keys():
                                response[idx] = str(belief_state['booking'][attr])
                                continue
                        if attr == 'place' and response[idx + 1].lower() == 'towninfo':
                            response[idx] = 'Cambridge'
                            continue
                        elif attr == 'area' and response[idx - 1].lower() == 'towninfo':
                            response[idx] = 'centre'
                            continue
                        else:
                            origin_attr = attr
                            if attr in value_replace.keys() and talk in value_replace[attr].keys():
                                if talk == 'train' and (attr == 'time' or attr == 'place'):
                                    if (talk + '_' +attr) not in slot_before.keys() and ('value_' +attr) not in slot_before.keys():
                                        attr = value_replace[attr][talk][0]
                                    else:
                                        attr = value_replace[attr][talk][1]
                                    slot_before[talk + '_' +attr].append(attr)
                                    slot_before['value_' +attr].append(attr)
                                else:
                                    attr = value_replace[attr][talk]
                            filled = get_value(talk, belief_state, attr, slot_before)
                            # logging.info(talk + ' -> ' + slot + ' -> ' + attr + ' -> ' + filled)
                            if filled:
                                response[idx] = str(filled)
                                if filled not in slot_before[talk + '_' + attr]:
                                    slot_before[talk + '_' + attr].append(filled)
                                continue
                            else:
                                for k in belief_state.keys():
                                    if k != 'booking':
                                        attr = origin_attr
                                        if origin_attr in value_replace.keys() and k in value_replace[origin_attr].keys():
                                            attr = value_replace[origin_attr][k]
                                        if attr in domain_attr_all[k]:
                                            filled = get_value(k, belief_state, attr, slot_before)
                                            if filled:
                                                response[idx] = str(filled)
                                                if filled not in slot_before[k + '_' + attr]:
                                                    slot_before[k + '_' + attr].append(filled)
                                                    continue
            if 'count' in response[idx]:
                response[idx] = str(3)
            # if 'place' in response[idx]:
            #     response[idx] = 'place'
            # if 'area' in response[idx]:
            #     response[idx] = 'area'
            # if 'price' in response[idx]:
            #     response[idx] = '2.3＄'
            # if 'pricerange' in response[idx]:
            #     response[idx] = 'cheap'
            # if 'time' in response[idx]:
            #     response[idx] = '12:30'
            if response[idx].startswith('[') and response[idx].endswith(']'):
                response[idx] = response[idx][1:-1]
                # logging.info(talk + ' -> ' + slot + ' -> ' + response[idx])
        return ' '.join(response).replace('[', '').replace(']', '').replace('value_', '')
    except:
        return ' '.join(response).replace('[', '').replace(']', '').replace('value_', '')


def get_value(domain, belief_state, attr, slot_before:defaultdict):
    try:
        if attr == 'reference':
            return reference
        if domain in belief_state.keys() and domain in domain_attr.keys():
            venues = db.queryResultVenues(domain, bs=belief_state[domain])
            if venues:
                if attr in domain_attr[domain]:
                    if slot_before[domain + '_' + attr]:
                        for idx in range(len(venues)):
                            if venues[idx][domain_attr[domain].index(attr)] not in slot_before[domain + '_' + attr]:
                                return venues[idx][domain_attr[domain].index(attr)]
                        return venues[0][domain_attr[domain].index(attr)]
                    else:
                        return venues[0][domain_attr[domain].index(attr)]
            else:
                venues = db.queryResultVenues(domain, bs={})
                if attr in domain_attr[domain]:
                    if slot_before[domain + '_' + attr]:
                        for idx in range(len(venues)):
                            if venues[idx][domain_attr[domain].index(attr)] not in slot_before[domain + '_' + attr]:
                                return venues[idx][domain_attr[domain].index(attr)]
                        return venues[0][domain_attr[domain].index(attr)]
                    else:
                        return venues[0][domain_attr[domain].index(attr)]
        elif domain == 'police' and attr in police.keys():
            return police[attr]
        elif domain == 'taxi' and attr in taxi_dict.keys():
            return taxi_dict[attr]
        elif domain == 'hospital' and domain in belief_state.keys():
            if attr == 'address':
                return '124 tenison road'
            if attr == 'name':
                return 'people hospital'
            if attr == 'postcode':
                return 'cb12dp'
            venues = db.queryResultVenues(domain, bs=belief_state[domain])
            if venues:
                if slot_before[domain + '_' + attr]:
                    for idx in range(len(venues)):
                        if attr in venues[idx].keys() and venues[idx][attr] not in slot_before[domain + '_' + attr]:
                            return venues[idx][attr]
                else:
                    if attr in venues[0].keys():
                        return venues[0][attr]
            else:
                venues = db.queryResultVenues(domain, bs={})
                if slot_before[domain + '_' + attr]:
                    for idx in range(len(venues)):
                        if attr in venues[idx].keys() and venues[idx][attr] not in slot_before[domain + '_' + attr]:
                            return venues[idx][attr]
                else:
                    if attr in venues[0].keys():
                        return venues[0][attr]
        return ''
    except:
        return ''

def get_talk(bs:dict, response:str):
    try:
        domain_num_bs = len(bs.keys())
        if 'booking' in bs.keys():
            domain_num_bs -= 1
        if domain_num_bs == 1:
            for k in bs.keys():
                if k != 'booking':
                    return k
        else:
            slot_list = []
            response = response.split()
            for idx in range(len(response)):
                if response[idx].startswith('[') and response[idx].endswith(']'):
                    slot = response[idx][1:-1]
                    slot_list.append(slot)
                    domain = slot.split('_')[0]
                    if domain in bs.keys() or domain in ['hospital', 'police']:
                        return domain
            for slot in slot_list:
                domain = slot.split('_')[0]
                attr = slot.split('_')[-1]
                if domain != 'value':
                    for k in bs.keys():
                        if k != 'booking' and attr in domain_attr_all[k]:
                            return k
            for slot in slot_list:
                attr = slot.split('_')[-1]
                for k in bs.keys():
                    if k != 'booking' and attr in domain_attr_all[k]:
                        return k
            for slot in slot_list:
                attr = slot.split('_')[-1]
                if attr in value_replace.keys():
                    for k in bs.keys():
                        if k in value_replace[attr].keys():
                            return k
            for domain in current_domain:
                if domain in bs.keys():
                    return domain
            for slot in slot_list:
                attr = slot.split('_')[-1]
                for domain in ['hospital', 'police']:
                    if attr in domain_attr_all[domain]:
                        return domain
            return 'hospital'
    except:
        return 'hospital'

def get_item(bs:dict):
    try:
        res = {}
        for domain in bs.keys():
            if domain in eval_domains:
                if domain == 'taxi':
                    res[domain] = taxi
                else:
                    eval_bs = {}
                    for slot in bs[domain].keys():
                        if slot in domain_attr_all[domain]:
                            eval_bs[slot] = bs[domain][slot]
                    venues = db.queryResultVenues(domain, bs=eval_bs)
                    random.shuffle(venues)
                    res[domain] = venues
        return res
    except:
        return {}

# text_list = [    "belief : restaurant food = dont care ; pricerange = expensive ; area = centre | booking time = 11:30 | train destination = cambridge ; day = thursday ; arriveby = 19:30 ; departure = cambridge | booking people = 2system : i would recommend [restaurant_name]. it is a [value_food] restaurant.",
# ]
# res, res_bs = parse_decoding_results_direct(text_list)
# print(text_list[0])
# print(get_talk(res_bs, res))