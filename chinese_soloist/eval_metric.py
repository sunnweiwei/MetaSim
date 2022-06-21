import json
from evaluation import eval_all
from tqdm import tqdm
def filter_domain(actions):
    a1 = ['配送周期', '物流全程跟踪', '联系配送', '配送方式', '返回方式', '预约配送时间', '拒收',
          '能否自提', '能否配送', '发错货', '下单地址填写', '发货检查']
    a5 = ['联系客服', '联系客户', '联系商家', '联系售后', '投诉']
    a = a1 + a5
    return all([act in a for act in actions if act != 'other']) and any([act in a1 for act in actions])

def eval_slot_match(predicts, answers, skip_null=True):
    match = []
    for pred, ans in zip(predicts, answers):
        if len(ans) == 0 and skip_null:
            continue
        match.append(int(all([slot[0] in pred for slot in ans])))
    return {'slot_match': sum(match) / len(match) * 100}

eval_systems = ['data-100_context-15', 'data-100_context-3', 'data-100_context-1', 'data-10_context-15', 'data-1_context-15']
# data = load_data('test_data_labels_entity.json')
output_labels = json.load(open('output/gt.json'))
test_data = open("data/test_set.txt", 'r', encoding='utf-8')
test_data_ent = list(open("data/test_set_jieba.txt", 'r', encoding='utf-8'))

index = 0
filter_index = []
end = False
ent_list = []
for dig_idx, line in tqdm(enumerate(test_data)):
    dialog_ent = json.loads(test_data_ent[dig_idx].strip('\n'))
    line.strip('\n')
    dialog = json.loads(line)
    goal = dialog["goal"]
    dialog_ind = False if len(goal['intents']) == 0 or not filter_domain(goal['intents']) else True
    current_turns = []
    for turn_idx, turn in enumerate(dialog['turns']):
        if turn['speaker'] == 'Q':
            uttrance = '用户：' + turn['turn']
        else:
            uttrance = '系统：' + turn['turn']
        if turn['speaker'] == 'A' and current_turns:
            if dialog_ind:
                if index >= len(output_labels):
                    end = True
                    break
                filter_index.append(index)
                print(dialog_ent[turn_idx]['ents'])
                ent_list.append(dialog_ent[turn_idx]['ents'])
            index += 1
        current_turns.append(uttrance)
    if end:
        break
    # print(ent_list)


for eval_system in eval_systems:
    output_predict = json.load(open(f'output/{eval_system}.json'))
    print(f"eval_system: {eval_system}")
    print(f"slot_acc: {eval_slot_match([output_predict[i] for i in filter_index], ent_list)}")
    # print(eval_all([output_predict[i] for i in filter_index], [output_labels[i] for i in filter_index]))