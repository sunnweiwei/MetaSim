import json

file_list = ['all_1.json', 'all_5.json', 'all_15.json']
for file in file_list:
    file_path = 'output/' + file
    data = json.load(open(file_path))
    ans = []
    for d in data:
        ans_tmp = []
        for text in d:
            ans_tmp.append(text.split('=>')[1])
        ans.append(ans_tmp)
    output_path = 'output/' + file.split('.')[0] + 'new.json'
    json.dump(ans, open(output_path, 'w'), indent=2)
