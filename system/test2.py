import os
import pickle
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

system_dict = defaultdict(list)
save_path = 'output/interaction/'
file_list = list(os.listdir(save_path))
for file in tqdm(file_list):
    if 'jddc' in file:
        task_name = file.split('_')[1]
        data = read_pkl(f"{save_path}{file}")
        system_dict[task_name].append(data)

save_path = 'interaction_generate/jddc/'
os.makedirs(save_path, exist_ok=True)
for key in system_dict.keys():
    write_pkl(system_dict[key], f"{save_path}{key}.json")

