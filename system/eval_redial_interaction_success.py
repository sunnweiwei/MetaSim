import os
import pickle
import re
from collections import defaultdict

from tqdm import tqdm

def write_pkl(obj, filename):
    dirname = '/'.join(filename.split('/')[:-1])
    os.makedirs(dirname, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def read_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

save_path = 'interaction_generate/redial/'
file_list = os.listdir(save_path)
success_rate = defaultdict(int)
for task in file_list:
    data = read_pkl(f"{save_path}{task}")
    for idx, dialogue in enumerate(data):
        context = dialogue['context']
        success = False
        mentioned_movie_id = []
        goal_movie_id = re.findall(r'@(\d+)', dialogue['goal'])
        for cxt in context:
            if not success:
                current_movie_id = re.findall(r'@(\d+)', cxt)
                if 'system:' in cxt:
                    if any([(id in goal_movie_id and id not in mentioned_movie_id) for id in current_movie_id]):
                        success = True
                mentioned_movie_id.extend(current_movie_id)
        data[idx]['success'] = success
        if success:
            success_rate[task] += 1
    success_rate[task] /= len(data)
    print(f"{task}: {success_rate[task]}")
    write_pkl(data, f"{save_path}{task}_new.json")