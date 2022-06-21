import json
import os
from collections import defaultdict,Counter

path = 'server/multiwoz_save/'

metrics = ['satisfaction', 'success', 'efficiency', 'naturalness']
info = defaultdict(dict)
for file in os.listdir(path):
    system = file.split('_')[1]
    data = json.load(open(path + file, 'r'))
    if 'count' not in info[system].keys():
        info[system]['count'] = 0
    info[system]['count'] += 1
    for metric in metrics:
        if metric not in info[system].keys():
            info[system][metric] = []
        info[system][metric].append(int(data[metric]))
for system in info.keys():
    info[system]['avg_score'] = 0
    for metric in metrics:
        info[system]['avg_score'] += sum(info[system][metric])
        info[system][metric] = Counter(info[system][metric])
for system in info.keys():
    info[system]['avg_score'] /= info[system]['count']
print(info)
rank = []
for system in info.keys():
    rank.append((info[system]['avg_score'], system))
print(sorted(rank, reverse=True))
# for system in info.keys():
#     for metric in metrics:


