import os

path = 'system/'
import json
# import time
#
# print(time.time())

# for idx in range(54):
#     path_sys = path + str(idx) + '/'
#     config = json.load(open(path_sys + 'config.json', 'r'))
#     print(idx)
#     print(config['system'])
#     print(config['task'])
#     print('\n\n')

for idx in range(18):
    path_sys = path + str(idx)
    os.makedirs(path_sys, exist_ok=True)
    input_path = path_sys + '/input'
    output_path = path_sys + '/output'
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    config = {}
    config['dataset'] = 'multiwoz'
    config['system'] = 'data-10_context-15'
    config['task'] = 'mwoz-domain-15'

    # # multiwoz context
    # if idx < 6:
    #     config['dataset'] = 'multiwoz'
    #     config['system'] = 'data-100_context-15'
    #     config['task'] = 'mwoz-context-15'
    # elif idx < 12:
    #     config['dataset'] = 'multiwoz'
    #     config['system'] = 'data-100_context-3'
    #     config['task'] = 'mwoz-context-3'
    # elif idx < 18:
    #     config['dataset'] = 'multiwoz'
    #     config['system'] = 'data-100_context-1'
    #     config['task'] = 'mwoz-context-1'
    # # multiwoz domain
    # elif idx < 24:
    #     config['dataset'] = 'multiwoz'
    #     config['system'] = 'data-10_context-15'
    #     config['task'] = 'mwoz-domain-3'
    # elif idx < 30:
    #     config['dataset'] = 'multiwoz'
    #     config['system'] = 'data-1_context-15'
    #     config['task'] = 'mwoz-domain-1'
    # # multiwoz recommender
    # elif idx < 36:
    #     config['dataset'] = 'multiwoz'
    #     config['system'] = 'data-100_context-15'
    #     config['task'] = 'mwoz-recommend-3'
    # elif idx < 42:
    #     config['dataset'] = 'multiwoz'
    #     config['system'] = 'data-100_context-15'
    #     config['task'] = 'mwoz-recommend-1'
    # # elif idx < 48:
    # #     config['dataset'] = 'multiwoz'
    # #     config['system'] = 'data-100_context-15'
    # #     config['task'] = 'redial-context-15'
    # else:
    #     config['dataset'] = 'multiwoz'
    #     config['system'] = 'data-100_context-15'
    #     config['task'] = 'mwoz-context-15'
    json.dump(config, open(path_sys + '/config.json', 'w'))



# for idx in range(54):
#     path_sys = path + str(idx) + '/'
#     config = json.load(open(path_sys + 'config.json', 'r'))
#     # multiwoz context
#     if idx < 6:
#         config['dataset'] = 'multiwoz'
#         config['system'] = 'data-100_context-15'
#         config['task'] = 'mwoz-context-15'
#     elif idx < 12:
#         config['dataset'] = 'multiwoz'
#         config['system'] = 'data-100_context-3'
#         config['task'] = 'mwoz-context-3'
#     elif idx < 18:
#         config['dataset'] = 'multiwoz'
#         config['system'] = 'data-100_context-1'
#         config['task'] = 'mwoz-context-1'
#     # multiwoz domain
#     elif idx < 24:
#         config['dataset'] = 'multiwoz'
#         config['system'] = 'data-10_context-15'
#         config['task'] = 'mwoz-domain-3'
#     elif idx < 30:
#         config['dataset'] = 'multiwoz'
#         config['system'] = 'data-1_context-15'
#         config['task'] = 'mwoz-domain-1'
#     # multiwoz recommender
#     elif idx < 36:
#         config['dataset'] = 'multiwoz'
#         config['system'] = 'data-100_context-15'
#         config['task'] = 'mwoz-recommend-3'
#     elif idx < 42:
#         config['dataset'] = 'multiwoz'
#         config['system'] = 'data-100_context-15'
#         config['task'] = 'mwoz-recommend-1'
    # redial base
    # elif idx < 48:
    #     config['dataset'] = 'multiwoz'
    #     config['system'] = 'data-100_context-15'
    #     config['task'] = 'redial-context-15'
    # JDDC base
    # else:
    #     config['dataset'] = 'multiwoz'
    #     config['system'] = 'data-100_context-15'
    #     config['task'] = 'mwoz-context-15'
    # json.dump(config, open(path_sys + 'config.json', 'w'))
#
# domain_attr = {
#     'attraction': ['id', 'address', 'area', 'entrance fee', 'name', 'phone', 'postcode', 'pricerange', 'openhours', 'type'],
#     'restaurant': ['id', 'address', 'area', 'food', 'introduction', 'name', 'phone', 'postcode', 'pricerange', 'location', 'type'],
#     'hotel': ['id', 'address', 'area', 'internet', 'parking', 'single', 'double', 'family', 'name', 'phone', 'postcode', 'pricerange', 'takesbookings', 'stars', 'type'],
#     'train': ['id', 'arriveBy', 'day', 'departure', 'destination', 'duration', 'leaveAt', 'price'],
#     'hospital': ['department', 'id', 'phone'],
# }
# from evaluate import MultiWozDB
# from collections import defaultdict
# db = MultiWozDB()
# slot_value = defaultdict(dict)
# for domain in domain_attr.keys():
#     for slot in domain_attr[domain]:
#         slot_value[domain][slot] = []
# for domain in domain_attr.keys():
#     venues = db.queryResultVenues(domain, bs={})
#     for venue in venues:
#         for id in range(len(domain_attr[domain])):
#             if venue[id] not in slot_value[domain][domain_attr[domain][id]]:
#                 slot_value[domain][domain_attr[domain][id]].append(venue[id])
# print(slot_value)

