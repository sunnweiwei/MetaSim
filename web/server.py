import datetime
import os
import random
import time

import requests
from flask import Flask, request, render_template, jsonify
from flask_cors import *
import json

app = Flask(__name__, template_folder='./static/templates')
CORS(app, supports_credentials=True)
from parser import parse_belief_state_all
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/', methods=["GET"])
def webchatindex():
    return render_template('index.html')


@app.route('/phone', methods=["GET"])
def webchatindex_phone():
    return render_template('phone_index.html')


goal_path = 'server/multiwoz_goal/'
save_path = 'server/multiwoz_save/'
assign_num = 5
task_id = 0
task_total = 3


@app.route('/goal', methods=["POST"])
def webchat():
    global goal_path, assign_num, task_id, task_total
    if not request.data:
        return ('fail')
    data = request.data.decode('utf-8')
    data = json.loads(data)
    logger.info(data)
    user = None
    if 'user' in data.keys():
        user = data['user']
    info = json.load(open(goal_path + 'info.json', 'r'))
    if user in info['user']:
        user_info_path = goal_path + user + '_info.json'
        user_goal_path = goal_path + user + '_goal.json'
        user_info = json.load(open(user_info_path, 'r'))
        user_goal = json.load(open(user_goal_path, 'r'))
    else:
        all_goal = json.load(open(goal_path + 'all_goal.json', 'r'))
        info['user'].append(user)
        assign_goal_list = [i for i in range(995 - (info['index'] + assign_num), 995 - info['index'])]
        info['index'] += assign_num
        info['assign'] += assign_goal_list
        user_info = {
            'index': 0,
            'finish': [],
            'assign': [],
        }
        user_info['assign'] += assign_goal_list
        user_goal = []

        if task_id == 0:
            for i in assign_goal_list:
                tmp = {
                    'system': 'mwoz-context-15',
                    'task': 'human',
                    'goal_index': i
                }
                for key in all_goal[i].keys():
                    tmp[key] = all_goal[i][key]
                user_goal.append(tmp)
            for i in assign_goal_list:
                tmp = {
                    'system': 'mwoz-context-3',
                    'task': 'human',
                    'goal_index': i
                }
                for key in all_goal[i].keys():
                    tmp[key] = all_goal[i][key]
                user_goal.append(tmp)
            for i in assign_goal_list:
                tmp = {
                    'system': 'mwoz-context-1',
                    'task': 'human',
                    'goal_index': i
                }
                for key in all_goal[i].keys():
                    tmp[key] = all_goal[i][key]
                user_goal.append(tmp)
        elif task_id == 1:
            for i in assign_goal_list:
                tmp = {
                    'system': 'mwoz-context-15',
                    'task': 'human',
                    'goal_index': i
                }
                for key in all_goal[i].keys():
                    tmp[key] = all_goal[i][key]
                user_goal.append(tmp)
            for i in assign_goal_list:
                tmp = {
                    'system': 'mwoz-recommend-3',
                    'task': 'human',
                    'goal_index': i
                }
                for key in all_goal[i].keys():
                    tmp[key] = all_goal[i][key]
                user_goal.append(tmp)
            for i in assign_goal_list:
                tmp = {
                    'system': 'mwoz-recommend-1',
                    'task': 'human',
                    'goal_index': i
                }
                for key in all_goal[i].keys():
                    tmp[key] = all_goal[i][key]
                user_goal.append(tmp)
        else:
            for i in assign_goal_list:
                tmp = {
                    'system': 'mwoz-context-15',
                    'task': 'human',
                    'goal_index': i
                }
                for key in all_goal[i].keys():
                    tmp[key] = all_goal[i][key]
                user_goal.append(tmp)
            for i in assign_goal_list:
                tmp = {
                    'system': 'mwoz-domain-3',
                    'task': 'human',
                    'goal_index': i
                }
                for key in all_goal[i].keys():
                    tmp[key] = all_goal[i][key]
                user_goal.append(tmp)
            for i in assign_goal_list:
                tmp = {
                    'system': 'mwoz-domain-1',
                    'task': 'human',
                    'goal_index': i
                }
                for key in all_goal[i].keys():
                    tmp[key] = all_goal[i][key]
                user_goal.append(tmp)

        task_id += 1
        if task_id >= task_total:
            task_id = 0
        random.shuffle(user_goal)
        json.dump(info, open(goal_path + 'info.json', 'w'))
        json.dump(user_info, open(goal_path + user + '_info.json', 'w'))
        json.dump(user_goal, open(goal_path + user + '_goal.json', 'w'))
    # if user_info['index'] > len(user_goal):
    #     data_send = {'goal': {}, 'goal_str': '', 'user': user, 'message': '',
    #                  'system': system,
    #                  'task': task}
    goal = user_goal[int(user_info['index'])]['goal']
    goal_str = '[ ' + parse_goal(goal) + ' ]'
    goal = parse_belief_state_all(goal_str)
    message = user_goal[int(user_info['index'])]['message']
    system = user_goal[int(user_info['index'])]['system']
    task = user_goal[int(user_info['index'])]['task']
    data_send = {'goal': goal, 'goal_str': goal_str, 'user': user, 'message': '<br>'.join(message), 'system': system,
                 'task': task}
    logger.info(data_send)
    return jsonify(data_send)


@app.route('/movie', methods=["POST"])
def webchatnmovie():
    if not request.data:  # 检测是否有数据
        return ('fail')
    data = request.data.decode('utf-8')
    # 获取到POST过来的数据，因为我这里传过来的数据需要转换一下编码。根据晶具体情况而定
    data = json.loads(data)
    logger.info(data)
    user = None
    if 'user' in data.keys():
        user = data['user']
    movie = [
        ["Spider-Man: No Way Home",
         "With Spider-Man's identity now revealed, Peter asks Doctor Strange for help. When a spell goes wrong, dangerous foes from other worlds start to appear, forcing Peter to discover what it truly means to be Spider-Man."],
        ["The Expanse",
         "In the 24th century, a group of humans untangle a vast plot which threatens the Solar System's fragile state of detente."],
        ["Encanto",
         "A young Colombian girl has to face the frustration of being the only member of her family without magical powers."],
        ["Scream",
         "Twenty-five years after the original series of murders in Woodsboro, a new Ghostface emerges, and Sidney Prescott must return to uncover the truth."],
        ["Don't Look Up",
         "Two low-level astronomers must go on a giant media tour to warn mankind of an approaching comet that will destroy planet Earth."],
        ["Spider-Man: No Way Home",
         "With Spider-Man's identity now revealed, Peter asks Doctor Strange for help. When a spell goes wrong, dangerous foes from other worlds start to appear, forcing Peter to discover what it truly means to be Spider-Man."],
        ["The Expanse",
         "In the 24th century, a group of humans untangle a vast plot which threatens the Solar System's fragile state of detente."],
        ["Encanto",
         "A young Colombian girl has to face the frustration of being the only member of her family without magical powers."],
        ["Scream",
         "Twenty-five years after the original series of murders in Woodsboro, a new Ghostface emerges, and Sidney Prescott must return to uncover the truth."],
        ["Don't Look Up",
         "Two low-level astronomers must go on a giant media tour to warn mankind of an approaching comet that will destroy planet Earth."],
    ]
    data_send = {'movie': movie, 'user': user}
    # 把区获取到的数据转为JSON格式。
    return jsonify(data_send)


def call_api(context, goal=None, cache=None, config=None):
    if len(context.strip()) == 0:
        return '', '{}'
    num_try = 0
    while True:
        try:
            if num_try > 100:
                return '', '{}', '', False
            url = "http://10.102.33.3:8081"
            if len(cache) == 0:
                cache = '{}'
            data = {"context": context, 'goal': goal, 'cache': cache, **config}
            res = requests.post(url=url, data=data, timeout=20)
            logger.info(res)
            res = json.loads(res.text)
            output = res['text']
            origin = res['origin']
            break
        except:
            num_try += 1
            logger.error('error')
            time.sleep(1)
    # output = input('>> ')
    system_key = 'system'
    if not output.startswith(f'{system_key}: '):
        output = f'{system_key}: ' + output
    cache = res['cache']
    # cache = '{}'
    return output, cache, origin, True


cache = {}


def parse_goal(goal):
    flat_goal = []
    for key in goal:
        if key in ['topic', 'message'] or len(goal[key]) == 0:
            continue
        att = []
        for cat in ['info', 'book']:
            if cat not in goal[key]:
                continue
            for k, v in goal[key][cat].items():
                # k = k.replace('_', ' ')
                att.append(f'{k} = {v}')
        bs = f'domain = {key} ; ' + ' ; '.join(att)
        flat_goal.append(bs)
    return ' | '.join(flat_goal).lower()


@app.route('/response', methods=["POST"])
def send_response():
    if not request.data:  # 检测是否有数据
        return ('fail')
    data = request.data.decode('utf-8')
    # 获取到POST过来的数据，因为我这里传过来的数据需要转换一下编码。根据晶具体情况而定
    data = json.loads(data)
    logger.info(data)
    context = data['context']
    goal = data["goal_str"]
    system_name = data['system']
    task_name = data['task']
    cache = data['cache']
    config = {'system': system_name, 'task': task_name}
    user = None
    if 'user' in data.keys():
        user = data['user']
    system_res, cache, origin_res, status = call_api(' [SEP] '.join(context), goal, cache, config)
    if not status:
        data_send = {'response': system_res, 'user': user, 'cache': cache, 'origin': origin_res, 'status': 0}
        return jsonify(data_send)
    system_res = system_res.replace(' -s', 's').replace(' -ly', 'ly')
    origin_res = origin_res.replace(' -s', 's').replace(' -ly', 'ly')
    data_send = {'response': system_res, 'user': user, 'cache': cache, 'origin': origin_res, 'status': 1}
    logger.info(data_send)
    # 把区获取到的数据转为JSON格式。
    return jsonify(data_send)

@app.route('/user', methods=["GET"])
def webchatindex_user():
    return render_template('user_info.html')

@app.route('/user_info', methods=["POST"])
def get_user_info():
    if not request.data:  # 检测是否有数据
        return ('fail')
    data = request.data.decode('utf-8')
    data = json.loads(data)
    logger.info(data)
    user = data['user']
    path = 'server/multiwoz_goal/'
    if not os.path.exists(path + user + '_info.json'):
        data_send = {'finish': 0, 'user': user, 'status': 0}
        return jsonify(data_send)
    user_info = json.load(open(path + user + '_info.json', 'r'))
    data_send = {'finish': len(user_info['finish']), 'user': user, 'status':1}
    logger.info(data_send)
    # 把区获取到的数据转为JSON格式。
    return jsonify(data_send)


@app.route('/save', methods=["POST"])
def save_dialogue():
    global goal_path, save_path, task_id, task_total
    # try:
    if not request.data:  # 检测是否有数据
        return ('fail')
    data = request.data.decode('utf-8')
    # 获取到POST过来的数据，因为我这里传过来的数据需要转换一下编码。根据晶具体情况而定
    data = json.loads(data)
    logger.info(data)
    context = data['context']
    origin_context = data['origin']
    goal = data["goal"]
    system_name = data['system']
    task_name = data['task']
    satisfaction = data['satisfaction']
    success = data['success']
    efficiency = data['efficiency']
    naturalness = data['naturalness']
    state = data['state']
    user = None
    if 'user' in data.keys():
        user = data['user']
    # user_goal_index_path = goal_path + user + '_index.txt'
    # with open(user_goal_index_path, 'r') as f:
    #     goal_index = int(f.readline().strip())
    info = json.load(open(goal_path + 'info.json', 'r'))
    user_info_path = goal_path + user + '_info.json'
    user_goal_path = goal_path + user + '_goal.json'
    user_info = json.load(open(user_info_path, 'r'))
    user_goal = json.load(open(user_goal_path, 'r'))
    goal_index = user_goal[user_info['index']]['goal_index']
    user_info['finish'].append(goal_index)
    info['finish'].append(goal_index)
    user_info['index'] += 1
    if user_info['index'] >= len(user_goal):
        all_goal = json.load(open(goal_path + 'all_goal.json', 'r'))
        assign_goal_list = [i for i in range(995 - (info['index'] + assign_num), 995 - info['index'])]
        info['index'] += assign_num
        info['assign'] += assign_goal_list
        user_info['assign'] += assign_goal_list
        user_info['index'] = 0
        user_goal = []

        if task_id == 0:
            for i in assign_goal_list:
                tmp = {
                    'system': 'mwoz-context-15',
                    'task': 'human',
                    'goal_index': i
                }
                for key in all_goal[i].keys():
                    tmp[key] = all_goal[i][key]
                user_goal.append(tmp)
            for i in assign_goal_list:
                tmp = {
                    'system': 'mwoz-context-3',
                    'task': 'human',
                    'goal_index': i
                }
                for key in all_goal[i].keys():
                    tmp[key] = all_goal[i][key]
                user_goal.append(tmp)
            for i in assign_goal_list:
                tmp = {
                    'system': 'mwoz-context-1',
                    'task': 'human',
                    'goal_index': i
                }
                for key in all_goal[i].keys():
                    tmp[key] = all_goal[i][key]
                user_goal.append(tmp)
        elif task_id == 1:
            for i in assign_goal_list:
                tmp = {
                    'system': 'mwoz-context-15',
                    'task': 'human',
                    'goal_index': i
                }
                for key in all_goal[i].keys():
                    tmp[key] = all_goal[i][key]
                user_goal.append(tmp)
            for i in assign_goal_list:
                tmp = {
                    'system': 'mwoz-recommend-3',
                    'task': 'human',
                    'goal_index': i
                }
                for key in all_goal[i].keys():
                    tmp[key] = all_goal[i][key]
                user_goal.append(tmp)
            for i in assign_goal_list:
                tmp = {
                    'system': 'mwoz-recommend-1',
                    'task': 'human',
                    'goal_index': i
                }
                for key in all_goal[i].keys():
                    tmp[key] = all_goal[i][key]
                user_goal.append(tmp)
        else:
            for i in assign_goal_list:
                tmp = {
                    'system': 'mwoz-context-15',
                    'task': 'human',
                    'goal_index': i
                }
                for key in all_goal[i].keys():
                    tmp[key] = all_goal[i][key]
                user_goal.append(tmp)
            for i in assign_goal_list:
                tmp = {
                    'system': 'mwoz-domain-3',
                    'task': 'human',
                    'goal_index': i
                }
                for key in all_goal[i].keys():
                    tmp[key] = all_goal[i][key]
                user_goal.append(tmp)
            for i in assign_goal_list:
                tmp = {
                    'system': 'mwoz-domain-1',
                    'task': 'human',
                    'goal_index': i
                }
                for key in all_goal[i].keys():
                    tmp[key] = all_goal[i][key]
                user_goal.append(tmp)

        task_id += 1
        if task_id >= task_total:
            task_id = 0
        random.shuffle(user_goal)
        json.dump(user_goal, open(goal_path + user + '_goal.json', 'w'))
    json.dump(info, open(goal_path + 'info.json', 'w'))
    json.dump(user_info, open(goal_path + user + '_info.json', 'w'))
    data_save = {
        'context': context,
        'origin_context': origin_context,
        'state': state,
        'goal': goal,
        'system': system_name,
        'task': task_name,
        'satisfaction': satisfaction,
        'success': success,
        'efficiency': efficiency,
        'naturalness': naturalness,
        'user': user,
        'goal_index': goal_index
    }
    logger.info('Save!')
    logger.info(data_save)
    json.dump(data_save,
              open(save_path + user + '_' + system_name + '_' + task_name + '_' + str(goal_index) + '.json', 'w'))
    # except:
    #     logger.error('Save error.')
    #     return jsonify({'stat': 1})
    logger.info('Save success.')
    return jsonify({'stat': 0})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='5000')
