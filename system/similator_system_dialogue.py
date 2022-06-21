import os
import json
from collections import defaultdict
from tqdm import tqdm
path = 'output/server/'

file_list = os.listdir(path)
dialogue = defaultdict(list)
for file in tqdm(file_list):
    task = file.split('_')[:-1]
    task = ''.join(task)
    # dialogue[task] += 1
    try:
        dialogue[task].append(json.load(open(path+file, 'r')))
    except:
        continue
for key in tqdm(dialogue.keys()):
    json.dump(dialogue[key], open('output/similator_system_dialogue/' + key + '.json', 'w'))