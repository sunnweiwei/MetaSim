import os
import sys

sys.path += ['./']
from mwoz_driver.train_mwoz import MWOZData, load_data, strip, lower, eval_action, write_file
from sklearn import tree, ensemble
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from collections import defaultdict
from sklearn import metrics
import numpy as np


# clf = tree.DecisionTreeClassifier()
def get_action(line):
    line = line.replace('user:', '').replace('system:', '')
    line = line.strip()
    return [item.split()[0].strip() for item in line.split('|')]


def remove_dup(line):
    out = []
    for i in line:
        if i not in out:
            out.append(i)
    return out


def train():
    clf = ensemble.RandomForestClassifier()
    # clf = GaussianNB()
    dataset = MWOZData(load_data('dataset/mwoz/MultiWOZ_2.1/train_data.json'), )

    slot_rank = defaultdict(list)
    slot_num = [0 for _ in range(100)]
    for goal, actions, ut in zip(dataset.goal, dataset.actions, dataset.response):
        if '[END]' not in ut:
            continue
        session_actions = []
        for action in actions:
            if 'user' not in action:
                continue
            act = action.replace('user:', '')
            act = [x.replace('inform', '').split(',') for x in act.split('|') if 'inform' in x]
            act = sum(act, [])
            act = [x.strip() for x in act if len(x.strip()) > 0]
            if 'inform' in action:
                slot_num[len(act)] += 1
            session_actions.extend(act)
        session_actions = remove_dup(session_actions)
        for i, slot in enumerate(session_actions):
            slot_rank[slot].append(i)
    slot_rank = {k: sum(v) / len(v) for k, v in slot_rank.items()}

    train_x = []
    train_y = []
    action_name = {}
    action_id_2_name = []
    for actions, goal in zip(dataset.actions, dataset.goal):
        x = get_action(actions[-2]) if len(actions) > 1 else ['none']
        y = get_action(actions[-1])
        train_x.append([])
        train_y.append([])
        for act in x:
            if act not in action_name:
                action_name[act] = len(action_name)
                action_id_2_name.append(act)
            train_x[-1].append(action_name[act])
        for act in y:
            if act not in action_name:
                action_name[act] = len(action_name)
                action_id_2_name.append(act)
            train_y[-1].append(action_name[act])

    train_data_x = []
    train_data_y = []
    for x, y in zip(train_x, train_y):
        train_data_x.append([0 for _ in range(len(action_name))])
        train_data_y.append([0 for _ in range(len(action_name))])
        # train_data_y.append(0)
        for i in x:
            train_data_x[-1][i] = 1
        for i in y:
            train_data_y[-1][i] = 1
            # train_data_y[-1] = i

    clf = clf.fit(X=train_data_x, y=train_data_y)
    print(action_id_2_name)

    # print(metrics.f1_score(train_data_y, clf.predict(train_data_x), average="macro"))
    print(metrics.f1_score(train_data_y, clf.predict(train_data_x), average="micro"))
    slot_num = [x / sum(slot_num) for x in slot_num]
    return clf, action_name, action_id_2_name, slot_rank, slot_num


def test(clf, action_name, action_id_2_name, slot_rank, slot_num):
    domain_rank = {'train': 10, 'taxi': 10}
    dataset = MWOZData(load_data('dataset/mwoz/MultiWOZ_2.1/test_data.json'), )

    test_x = []
    test_y = []
    for actions, goal in zip(dataset.actions, dataset.goal):
        x = get_action(actions[-2]) if len(actions) > 1 else ['none']
        y = get_action(actions[-1])
        test_x.append([])
        test_y.append([])
        for act in x:
            test_x[-1].append(action_name[act])
        for act in y:
            test_y[-1].append(action_name[act])

    test_data_x = []
    test_data_y = []
    for x, y in zip(test_x, test_y):
        test_data_x.append([0 for _ in range(len(action_name))])
        test_data_y.append([0 for _ in range(len(action_name))])

        for i in x:
            test_data_x[-1][i] = 1
        for i in y:
            test_data_y[-1][i] = 1

    def parse_act(line):
        return {item.split()[0]: strip(' '.join(item.split()[1:]).split(',')) for item in lower(line).split(' | ')}

    pred_data_y = clf.predict(test_data_x)
    output_pred = []
    for actions, goal, pred in zip(dataset.actions, dataset.goal, pred_data_y):
        if len(actions) == 1:
            actions = ['system: none'] + actions
        x = parse_act(actions[-2])
        y = parse_act(actions[-1])
        informed_slot = []
        for t in range(3, len(actions) + 1, 2):
            this_act = parse_act(actions[-t])
            if 'inform' in this_act:
                informed_slot.extend(this_act['inform'])
        pred_y = []
        for i, v in enumerate(pred):
            if v > 0:
                pred_y.append(action_id_2_name[i])
        # pred_y = [pred]
        if len(pred_y) == 0:
            pred_y.append('none')
        goal = goal.replace('[', '').replace(']', '')
        domain = [[s[:s.index('=')].strip() if 'domain' not in s else s[s.index('=') + 1:].strip()
                   for s in item.split(';') if 'domain' not in s] for item in goal.split('|')]
        domain.sort(key=lambda xx: domain_rank.get(xx[0], 0))
        domain = [sorted(item[1:], key=lambda xx: slot_rank.get(xx, 10)) for item in domain]
        domain = [d for d in sum(domain, []) if d not in ['domain', 'invalid']]

        text = 'user: '
        out = []
        if 'request' in x and 'inform' not in pred_y:
            pred_y.append('inform')
        for act in pred_y:
            if act == 'inform':
                can_slot = []
                if 'request' in x:
                    can_slot.extend(x['request'])
                for slot in domain:
                    if slot in informed_slot:
                        for ll in range(len(informed_slot)):
                            if informed_slot[ll] == slot:
                                informed_slot[ll] = '='
                                break
                    else:
                        can_slot.append(slot)
                can_slot = can_slot[:np.random.choice([ii for ii in range(len(slot_num))], p=slot_num)]
                out.append(f"{act} {' , '.join(can_slot)}")
            else:
                if act == 'end':
                    act = 'end of dialogue'
                out.append(f'{act}')
        text = text + ' | '.join(out)
        output_pred.append(text)
    return output_pred, [item[-1] for item in dataset.actions]


class AgendaPolicy:
    def __init__(self):
        self.clf, self.action_name, self.action_id_2_name, self.slot_rank, self.slot_num = train()
        self.slot_map = {'price': 'pricerange', 'depart': 'departure', 'dest': 'destination',
                         'leave': 'leaveat', 'arrive': 'arriveby'}
        self.domain_rank = {'train': 10, 'taxi': 10}

    def generate(self, dataset_actions, dataset_goal):
        dataset_actions = [dataset_actions]
        dataset_goal = [dataset_goal]
        test_x = []
        test_y = []
        for actions, goal in zip(dataset_actions, dataset_goal):
            x = get_action(actions[-2]) if len(actions) > 1 else ['none']
            y = get_action(actions[-1])
            test_x.append([])
            test_y.append([])
            for act in x:
                test_x[-1].append(self.action_name.get(act, 0))
            for act in y:
                test_y[-1].append(self.action_name.get(act, 0))

        test_data_x = []
        test_data_y = []
        for x, y in zip(test_x, test_y):
            test_data_x.append([0 for _ in range(len(self.action_name))])
            test_data_y.append([0 for _ in range(len(self.action_name))])
            for i in x:
                test_data_x[-1][i] = 1
            for i in y:
                test_data_y[-1][i] = 1

        def parse_act(line):
            return {item.split()[0]: strip(' '.join(item.split()[1:]).split(',')) for item in lower(line).split(' | ')}

        pred_data_y = self.clf.predict(test_data_x)
        output_pred = []
        for actions, goal, pred in zip(dataset_actions, dataset_goal, pred_data_y):
            if len(actions) == 1:
                actions = ['system: none'] + actions
            x = parse_act(actions[-2])
            y = parse_act(actions[-1])
            informed_slot = []
            for t in range(3, len(actions) + 1, 2):
                this_act = parse_act(actions[-t])
                if 'inform' in this_act:
                    informed_slot.extend(this_act['inform'])
            pred_y = []
            for i, v in enumerate(pred):
                if v > 0:
                    pred_y.append(self.action_id_2_name[i])
            if len(pred_y) == 0:
                pred_y.append('none')
            goal = goal.replace('[', '').replace(']', '')
            domain = [[s[:s.index('=')].strip() if 'domain' not in s else s[s.index('=') + 1:].strip()
                       for s in item.split(';') if 'domain' not in s] for item in goal.split('|')]
            domain.sort(key=lambda xx: self.domain_rank.get(xx[0], 0))
            domain = [sorted(item[1:], key=lambda xx: self.slot_rank.get(xx, 10)) for item in domain]
            domain = [d for d in sum(domain, []) if d not in ['domain', 'invalid']]

            text = 'user: '
            out = []
            if 'request' in x and 'inform' not in pred_y:
                pred_y.append('inform')
            for act in pred_y:
                if act == 'inform':
                    can_slot = []
                    if 'request' in x:
                        can_slot.extend(x['request'])
                    for slot in domain:
                        if slot in informed_slot:
                            for ll in range(len(informed_slot)):
                                if informed_slot[ll] == slot:
                                    informed_slot[ll] = '='
                                    break
                        else:
                            can_slot.append(slot)
                    can_slot = can_slot[:np.random.choice([ii for ii in range(len(self.slot_num))], p=self.slot_num)]
                    out.append(f"{act} {' , '.join(can_slot)}")
                else:
                    if act == 'end':
                        act = 'end of dialogue'
                    out.append(f'{act}')
            text = text + ' | '.join(out)
            output_pred.append(text)
        return output_pred[0]


def main():
    clf, action_name, action_id_2_name, slot_rank, slot_num = train()
    output_predict, _actions = test(clf, action_name, action_id_2_name, slot_rank, slot_num)

    print(eval_action(lower(output_predict), lower(_actions)))
    os.makedirs(f'ckpt/mwoz-agenda/text', exist_ok=True)
    write_file(output_predict, f'ckpt/mwoz-agenda/text/0.txt')


if __name__ == '__main__':
    main()
