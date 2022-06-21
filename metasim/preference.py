import sys

sys.path += ['./']
import json
from tqdm import tqdm
import numpy as np
import os
import sqlite3
import random

domin_list = ['taxi', 'police', 'hospital', 'hotel', 'attraction', 'train', 'restaurant']


def preprocessing(delex_data_path):
    print('Start preprocessing...')
    delex_dialogues = json.load(open(f'{delex_data_path}'))
    total = 0
    domin_total = {'taxi': 0, 'police': 0, 'hospital': 0, 'hotel': 0, 'attraction': 0, 'train': 0, 'restaurant': 0}
    fail_info_domin_total = {'taxi': 0, 'police': 0, 'hospital': 0, 'hotel': 0, 'attraction': 0, 'train': 0,
                             'restaurant': 0}
    reqt_domin_total = {'taxi': 0, 'police': 0, 'hospital': 0, 'hotel': 0, 'attraction': 0, 'train': 0, 'restaurant': 0}
    book_domin_total = {'taxi': 0, 'police': 0, 'hospital': 0, 'hotel': 0, 'attraction': 0, 'train': 0, 'restaurant': 0}
    book_fail_total = {'taxi': 0, 'police': 0, 'hospital': 0, 'hotel': 0, 'attraction': 0, 'train': 0, 'restaurant': 0}
    domin_combine_list = []
    domin_combine_dict = {}
    info_domin_atrr_combine_list = {'taxi': [], 'police': [], 'hospital': [], 'hotel': [], 'attraction': [],
                                    'train': [], 'restaurant': []}
    info_domin_atrr_combine_dict = {'taxi': {}, 'police': {}, 'hospital': {}, 'hotel': {}, 'attraction': {},
                                    'train': {}, 'restaurant': {}}
    fail_info_domin_atrr_list = {'taxi': [], 'police': [], 'hospital': [], 'hotel': [], 'attraction': [], 'train': [],
                                 'restaurant': []}
    fail_info_domin_atrr_dict = {'taxi': {}, 'police': {}, 'hospital': {}, 'hotel': {}, 'attraction': {}, 'train': {},
                                 'restaurant': {}}
    reqt_domin_atrr_combine_list = {'taxi': [], 'police': [], 'hospital': [], 'hotel': [], 'attraction': [],
                                    'train': [], 'restaurant': []}
    reqt_domin_atrr_combine_dict = {'taxi': {}, 'police': {}, 'hospital': {}, 'hotel': {}, 'attraction': {},
                                    'train': {}, 'restaurant': {}}
    book_domin_atrr_combine_list = {'taxi': [], 'police': [], 'hospital': [], 'hotel': [], 'attraction': [],
                                    'train': [], 'restaurant': []}
    book_domin_atrr_combine_dict = {'taxi': {}, 'police': {}, 'hospital': {}, 'hotel': {}, 'attraction': {},
                                    'train': {}, 'restaurant': {}}
    book_domin_atrr_value_list = {'taxi': {}, 'police': {}, 'hospital': {}, 'hotel': {}, 'attraction': {}, 'train': {},
                                  'restaurant': {}}
    book_domin_atrr_value_dict = {'taxi': {}, 'police': {}, 'hospital': {}, 'hotel': {}, 'attraction': {}, 'train': {},
                                  'restaurant': {}}
    fail_book_domin_atrr_list = {'taxi': [], 'police': [], 'hospital': [], 'hotel': [], 'attraction': [], 'train': [],
                                 'restaurant': []}
    fail_book_domin_atrr_dict = {'taxi': {}, 'police': {}, 'hospital': {}, 'hotel': {}, 'attraction': {}, 'train': {},
                                 'restaurant': {}}
    taxi_value_list = {'leaveAt': [], 'arriveBy': [], 'destination': [], 'departure': []}
    taxi_value_dict = {'leaveAt': {}, 'arriveBy': {}, 'destination': {}, 'departure': {}}
    for _, dialogue in tqdm(delex_dialogues.items(), desc='dialogues'):
        total += 1
        goal = dialogue['goal']

        # domin组合及概率
        domin_combine = []
        for domin, value in goal.items():
            if domin in domin_list and value:
                domin_total[domin] += 1
                domin_combine.append(domin)
        domin_combine = tuple(domin_combine)
        if domin_combine not in domin_combine_list:
            domin_combine_list.append(domin_combine)
            domin_combine_dict[domin_combine] = 1
        else:
            domin_combine_dict[domin_combine] += 1

        for domin in domin_combine:
            # inform{domin.attr}组合及概率
            if 'info' in goal[domin].keys():
                info = goal[domin]['info']
                info_attr_combine = []
                for attr, _ in info.items():
                    info_attr_combine.append(attr)
                info_attr_combine = tuple(info_attr_combine)
                if info_attr_combine not in info_domin_atrr_combine_list[domin]:
                    info_domin_atrr_combine_list[domin].append(info_attr_combine)
                    info_domin_atrr_combine_dict[domin][info_attr_combine] = 1
                else:
                    info_domin_atrr_combine_dict[domin][info_attr_combine] += 1

                if domin == 'taxi':
                    for attr, value in info.items():
                        if value not in taxi_value_list[attr]:
                            taxi_value_list[attr].append(value)
                            taxi_value_dict[attr][value] = 1
                        else:
                            taxi_value_dict[attr][value] += 1

            if 'fail_info' in goal[domin].keys() and goal[domin]['fail_info']:
                # fail_inform概率
                fail_info_domin_total[domin] += 1

                # fail_inform{domin.attr}概率
                fail_info = goal[domin]['fail_info']
                for attr, value in info.items():
                    if attr in fail_info.keys() and fail_info[attr] != value:
                        if attr not in fail_info_domin_atrr_list[domin]:
                            fail_info_domin_atrr_list[domin].append(attr)
                            fail_info_domin_atrr_dict[domin][attr] = 1
                        else:
                            fail_info_domin_atrr_dict[domin][attr] += 1

            if 'reqt' in goal[domin].keys() and goal[domin]['reqt']:
                # reqt概率
                reqt_domin_total[domin] += 1

                # reqt{domin.attr}概率
                reqt = tuple(goal[domin]['reqt'])
                if reqt not in reqt_domin_atrr_combine_list[domin]:
                    reqt_domin_atrr_combine_list[domin].append(reqt)
                    reqt_domin_atrr_combine_dict[domin][reqt] = 1
                else:
                    reqt_domin_atrr_combine_dict[domin][reqt] += 1

            if 'book' in goal[domin].keys() and goal[domin]['book']:
                # book概率
                book_domin_total[domin] += 1

                # domin.key取值概率
                book = goal[domin]['book']
                book_attr_combine = []
                for attr, value in book.items():
                    book_attr_combine.append(attr)
                    if attr not in book_domin_atrr_value_list[domin]:
                        book_domin_atrr_value_list[domin][attr] = []
                        book_domin_atrr_value_list[domin][attr].append(value)
                        book_domin_atrr_value_dict[domin][attr] = {}
                        book_domin_atrr_value_dict[domin][attr][value] = 1
                    else:
                        if value not in book_domin_atrr_value_list[domin][attr]:
                            book_domin_atrr_value_list[domin][attr].append(value)
                            book_domin_atrr_value_dict[domin][attr][value] = 1
                        else:
                            book_domin_atrr_value_dict[domin][attr][value] += 1
                # book{domin.attr}概率
                book_attr_combine = tuple(book_attr_combine)
                if book_attr_combine not in book_domin_atrr_combine_list[domin]:
                    book_domin_atrr_combine_list[domin].append(book_attr_combine)
                    book_domin_atrr_combine_dict[domin][book_attr_combine] = 1

                else:
                    book_domin_atrr_combine_dict[domin][book_attr_combine] += 1

                if 'fail_book' in goal[domin].keys() and goal[domin]['fail_book']:
                    # fail_book概率
                    book_fail_total[domin] += 1

                    # fail_book{domin.key}概率
                    fail_book = goal[domin]['fail_book']
                    for attr, value in fail_book.items():
                        if attr not in fail_book_domin_atrr_list[domin]:
                            fail_book_domin_atrr_list[domin].append(attr)
                            fail_book_domin_atrr_dict[domin][attr] = 1
                        else:
                            fail_book_domin_atrr_dict[domin][attr] += 1
    print('Done.')
    # print('total: ', total)
    # print('domin_total: ', domin_total)
    # print('fail_info_domin_total: ', fail_info_domin_total)
    # print('reqt_domin_total: ', reqt_domin_total)
    # print('book_domin_total: ', book_domin_total)
    # print('book_fail_total: ', book_fail_total)
    # print('domin_combine_list: ', domin_combine_list)
    # print('domin_combine_dict: ', domin_combine_dict)
    # print('info_domin_atrr_combine_list: ', info_domin_atrr_combine_list)
    # print('info_domin_atrr_combine_dict: ', info_domin_atrr_combine_dict)
    # print('fail_info_domin_atrr_list: ', fail_info_domin_atrr_list)
    # print('fail_info_domin_atrr_dict: ', fail_info_domin_atrr_dict)
    # print('reqt_domin_atrr_combine_list: ', reqt_domin_atrr_combine_list)
    # print('reqt_domin_atrr_combine_dict: ', reqt_domin_atrr_combine_dict)
    # print('book_domin_atrr_combine_list: ', book_domin_atrr_combine_list)
    # print('book_domin_atrr_combine_dict: ', book_domin_atrr_combine_dict)
    # print('book_domin_atrr_value_list: ', book_domin_atrr_value_list)
    # print('book_domin_atrr_value_dict: ', book_domin_atrr_value_dict)
    # print('fail_book_domin_atrr_list: ', fail_book_domin_atrr_list)
    # print('fail_book_domin_atrr_dict: ', fail_book_domin_atrr_dict)

    # 计算概率
    domin_combine_list_p = []
    for i in domin_combine_list:
        domin_combine_list_p.append(domin_combine_dict[i] / total)

    info_domin_atrr_combine_list_p = {'taxi': [], 'police': [], 'hospital': [], 'hotel': [], 'attraction': [],
                                      'train': [], 'restaurant': []}
    for domin, attrs in info_domin_atrr_combine_list.items():
        for attr in attrs:
            info_domin_atrr_combine_list_p[domin].append(info_domin_atrr_combine_dict[domin][attr] / domin_total[domin])

    fail_info_domin_total_p = {'taxi': 0, 'police': 0, 'hospital': 0, 'hotel': 0, 'attraction': 0, 'train': 0,
                               'restaurant': 0}
    for domin, num in fail_info_domin_total.items():
        fail_info_domin_total_p[domin] = num / domin_total[domin]

    fail_info_domin_atrr_list_p = {'taxi': [], 'police': [], 'hospital': [], 'hotel': [], 'attraction': [], 'train': [],
                                   'restaurant': []}
    for domin, attrs in fail_info_domin_atrr_list.items():
        total = 0
        for attr in attrs:
            total += fail_info_domin_atrr_dict[domin][attr]
        for attr in attrs:
            fail_info_domin_atrr_list_p[domin].append(fail_info_domin_atrr_dict[domin][attr] / total)

    reqt_domin_total_p = {'taxi': 0, 'police': 0, 'hospital': 0, 'hotel': 0, 'attraction': 0, 'train': 0,
                          'restaurant': 0}
    for domin, num in reqt_domin_total.items():
        reqt_domin_total_p[domin] = num / domin_total[domin]

    reqt_domin_atrr_combine_list_p = {'taxi': [], 'police': [], 'hospital': [], 'hotel': [], 'attraction': [],
                                      'train': [], 'restaurant': []}
    for domin, attrs in reqt_domin_atrr_combine_list.items():
        for attr in attrs:
            reqt_domin_atrr_combine_list_p[domin].append(
                reqt_domin_atrr_combine_dict[domin][attr] / reqt_domin_total[domin])

    book_domin_total_p = {'taxi': 0, 'police': 0, 'hospital': 0, 'hotel': 0, 'attraction': 0, 'train': 0,
                          'restaurant': 0}
    for domin, num in book_domin_total.items():
        book_domin_total_p[domin] = num / domin_total[domin]

    book_domin_atrr_combine_list_p = {'taxi': [], 'police': [], 'hospital': [], 'hotel': [], 'attraction': [],
                                      'train': [], 'restaurant': []}
    for domin, attrs in book_domin_atrr_combine_list.items():
        for attr in attrs:
            book_domin_atrr_combine_list_p[domin].append(
                book_domin_atrr_combine_dict[domin][attr] / book_domin_total[domin])

    book_domin_atrr_value_list_p = {'taxi': {}, 'police': {}, 'hospital': {}, 'hotel': {}, 'attraction': {},
                                    'train': {}, 'restaurant': {}}
    for domin, attrs in book_domin_atrr_value_list.items():
        for attr, value in attrs.items():
            if attr not in book_domin_atrr_value_list_p[domin].keys():
                book_domin_atrr_value_list_p[domin][attr] = []
            value_total = 0
            for v in value:
                value_total += book_domin_atrr_value_dict[domin][attr][v]
            for v in value:
                book_domin_atrr_value_list_p[domin][attr].append(
                    book_domin_atrr_value_dict[domin][attr][v] / value_total)

    book_fail_total_p = {'taxi': 0, 'police': 0, 'hospital': 0, 'hotel': 0, 'attraction': 0, 'train': 0,
                         'restaurant': 0}
    for domin, num in book_fail_total.items():
        book_fail_total_p[domin] = num / domin_total[domin]

    fail_book_domin_atrr_list_p = {'taxi': [], 'police': [], 'hospital': [], 'hotel': [], 'attraction': [], 'train': [],
                                   'restaurant': []}
    for domin, attrs in fail_book_domin_atrr_list.items():
        for attr in attrs:
            fail_book_domin_atrr_list_p[domin].append(fail_book_domin_atrr_dict[domin][attr] / book_fail_total[domin])

    taxi_value_list_p = {'leaveAt': [], 'arriveBy': [], 'destination': [], 'departure': []}
    for attr, values in taxi_value_list.items():
        total = 0
        for value in values:
            total += taxi_value_dict[attr][value]
        for value in values:
            taxi_value_list_p[attr].append(taxi_value_dict[attr][value] / total)

    # print('domin_combine_list_p: ', domin_combine_list_p)
    # print('info_domin_atrr_combine_list_p: ', info_domin_atrr_combine_list_p)
    # print('fail_info_domin_atrr_list_p: ', fail_info_domin_atrr_list_p)
    # print('reqt_domin_atrr_combine_list_p: ', reqt_domin_atrr_combine_list_p)
    # print('book_domin_atrr_combine_list_p: ', book_domin_atrr_combine_list_p)
    # print('book_domin_atrr_value_list_p: ', book_domin_atrr_value_list_p)
    # print('fail_book_domin_atrr_list_p: ', fail_book_domin_atrr_list_p)

    return domin_combine_list, domin_combine_list_p, info_domin_atrr_combine_list, info_domin_atrr_combine_list_p \
        , fail_info_domin_atrr_list, fail_info_domin_atrr_list_p, reqt_domin_atrr_combine_list \
        , reqt_domin_atrr_combine_list_p, book_domin_atrr_combine_list, book_domin_atrr_combine_list_p \
        , book_domin_atrr_value_list, book_domin_atrr_value_list_p, fail_book_domin_atrr_list \
        , fail_book_domin_atrr_list_p, fail_info_domin_total_p, reqt_domin_total_p, book_domin_total_p \
        , book_fail_total_p, taxi_value_list, taxi_value_list_p


def get_database(db_path):
    print('Load database...')
    db = {}
    db_column = {}
    for domin in domin_list:
        if domin == 'taxi' or domin == 'police' or domin == 'hospital':
            db[domin] = []
            fin = open(db_path + domin + '_db.json')
            db_json = json.load(fin)
            fin.close()
            db_column[domin] = [attr for attr in db_json[0].keys()]
            for entity in db_json:
                entity_info = []
                for _, v in entity.items():
                    entity_info.append(v)
                entity_info = tuple(entity_info)
                db[domin].append(entity_info)
            continue

        db_path = os.path.join('db/{}-dbase.db'.format(domin))
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        sql_query = "select * from {}".format(domin)
        items = c.execute(sql_query).fetchall()
        db_column[domin] = [tuple[0] for tuple in c.description]
        db[domin] = items
    print('Load database done.')
    return db, db_column


def get_other_item(ori_item, attr, db, dbc):
    new_item = None
    for item in db:
        flag = True
        for k, v in ori_item.items():
            if (k != attr and v != item[dbc.index(k)]) or (k == attr and v == item[dbc.index(k)]):
                flag = False
                break
        if flag:
            new_item = {}
            for k, _ in ori_item.items():
                new_item[k] = item[dbc.index(k)]
            break
    return new_item


def generate_goal(d, dp, id, idp, fid, fidp, rd, rdp, bd, bdp, bv, bvp, fbd, fbdp, fp, rp, bp, bfp, t, tp, db, dbc,
                  real=True):
    goal = {'taxi': {}, 'police': {}, 'hospital': {}, 'hotel': {}, 'attraction': {}, 'train': {}, 'restaurant': {}}
    domin_combine = d[np.random.choice(len(d), 1, p=dp)[0]]
    for domin in domin_combine:
        add_info = False
        goal[domin]['info'] = {}
        if domin == 'taxi':
            attr_combine = id[domin][np.random.choice(len(id[domin]), 1, p=idp[domin])[0]]
            for attr in attr_combine:
                goal[domin]['info'][attr] = t[attr][np.random.choice(len(t[attr]), 1, p=tp[attr])[0]]

            goal[domin]['reqt'] = []
            reqt_attr_combine = rd[domin][np.random.choice(len(rd[domin]), 1, p=rdp[domin])[0]]
            for attr in reqt_attr_combine:
                goal[domin]['reqt'].append(attr)
            continue
        item = db[domin][np.random.choice(len(db[domin]), 1)[0]]
        if real:
            attr_combine = id[domin][np.random.choice(len(id[domin]), 1, p=idp[domin])[0]]
        else:
            choose_num = random.randint(1, len(dbc[domin]))
            attr_combine = np.random.choice(dbc[domin], choose_num)
        for attr in attr_combine:
            goal[domin]['info'][attr] = item[dbc[domin].index(attr)]
        if domin in ['hotel', 'restaurant'] and len(attr_combine) == 1 and attr_combine[0] == 'name':
            add_info = True

        is_fail_info = np.random.choice([0, 1], 1, p=[1 - fp[domin], fp[domin]])[0]
        if is_fail_info == 1:
            new_item = None
            if real:
                info_attr_list = []
                info_attr_list_p = []
                total = 0
                for attr in attr_combine:
                    if attr in fid[domin]:
                        total += fidp[domin][fid[domin].index(attr)]
                for attr in attr_combine:
                    if attr in fid[domin]:
                        info_attr_list.append(attr)
                        info_attr_list_p.append(fidp[domin][fid[domin].index(attr)] / total)
                if len(info_attr_list) > 0:
                    fail_info_attr = info_attr_list[np.random.choice(len(info_attr_list), 1, p=info_attr_list_p)[0]]
                    new_item = get_other_item(goal[domin]['info'], fail_info_attr, db[domin], dbc[domin])
            else:
                if len(attr_combine) > 1:
                    fail_attr_index = np.random.choice(len(attr_combine), 1)[0]
                    new_item = get_other_item(goal[domin]['info'], attr_combine[fail_attr_index], db[domin], dbc[domin])
            if new_item:
                goal[domin]['fail_info'] = {}
                for k, v in new_item.items():
                    goal[domin]['fail_info'][k] = v

        is_reqt = np.random.choice([0, 1], 1, p=[1 - rp[domin], rp[domin]])[0]
        if is_reqt == 1 or add_info:
            if real:
                reqt_attr_list = []
                reqt_attr_list_p = []
                total = 0
                for attrs in rd[domin]:
                    flag = True
                    for attr in attrs:
                        if attr in goal[domin]['info'].keys():
                            flag = False
                            break
                    if flag:
                        reqt_attr_list.append(attrs)
                        total += rdp[domin][rd[domin].index(attrs)]
                for attrs in reqt_attr_list:
                    reqt_attr_list_p.append(rdp[domin][rd[domin].index(attrs)] / total)
                if len(reqt_attr_list) > 0:
                    goal[domin]['reqt'] = []
                    reqt_attr_combine = reqt_attr_list[np.random.choice(len(reqt_attr_list), 1, p=reqt_attr_list_p)[0]]
                    for attr in reqt_attr_combine:
                        goal[domin]['reqt'].append(attr)
            else:
                candidate = []
                for attr in dbc[domin]:
                    if attr not in attr_combine:
                        candidate.append(attr)
                if len(candidate) > 0:
                    goal[domin]['reqt'] = []
                    choose_num = random.randint(1, len(candidate))
                    reqt_attr_combine = np.random.choice(candidate, choose_num)
                    for attr in reqt_attr_combine:
                        goal[domin]['reqt'].append(attr)

        is_book = np.random.choice([0, 1], 1, p=[1 - bp[domin], bp[domin]])[0]
        if is_book == 1 or add_info:
            goal[domin]['book'] = {}
            if real:
                book_attr_combine = bd[domin][np.random.choice(len(bd[domin]), 1, p=bdp[domin])[0]]
            else:
                list_book_attr = []
                for attrs in bd[domin]:
                    for attr in attrs:
                        if attr not in list_book_attr:
                            list_book_attr.append(attr)
                choose_num = random.randint(1, len(list_book_attr))
                book_attr_combine = tuple(np.random.choice(list_book_attr, choose_num))
            for attr in book_attr_combine:
                if real:
                    value = bv[domin][attr][np.random.choice(len(bv[domin][attr]), 1, p=bvp[domin][attr])[0]]
                else:
                    value = bv[domin][attr][np.random.choice(len(bv[domin][attr]), 1)[0]]
                goal[domin]['book'][attr] = value
            is_fail_book = np.random.choice([0, 1], 1, p=[1 - bfp[domin], bfp[domin]])[0]
            if is_fail_book == 1:
                book_attr_list = []
                if real:
                    book_attr_list_p = []
                    total = 0
                    for attr in book_attr_combine:
                        if attr in fbd[domin]:
                            total += fbdp[domin][fbd[domin].index(attr)]
                    for attr in book_attr_combine:
                        if attr in fbd[domin]:
                            book_attr_list.append(attr)
                            book_attr_list_p.append(fbdp[domin][fbd[domin].index(attr)] / total)
                    fail_attr = book_attr_list[np.random.choice(len(book_attr_list), 1, p=book_attr_list_p)[0]]
                else:
                    for attr in book_attr_combine:
                        book_attr_list.append(attr)
                    fail_attr = np.random.choice(book_attr_list, 1)[0]

                if len(bv[domin][fail_attr]) > 1:
                    value_list = []
                    if real:
                        value_list_p = []
                        total = 0
                        for v in bv[domin][fail_attr]:
                            if v != goal[domin]['book'][fail_attr]:
                                total += bvp[domin][fail_attr][bv[domin][fail_attr].index(v)]
                        for v in bv[domin][fail_attr]:
                            if v != goal[domin]['book'][fail_attr]:
                                value_list.append(v)
                                value_list_p.append(bvp[domin][fail_attr][bv[domin][fail_attr].index(v)] / total)
                        value = value_list[np.random.choice(len(value_list), 1, p=value_list_p)[0]]
                    else:
                        for v in bv[domin][fail_attr]:
                            if v != goal[domin]['book'][fail_attr]:
                                value_list.append(v)
                        value = np.random.choice(value_list, 1)[0]
                    goal[domin]['fail_book'] = {}
                    goal[domin]['fail_book'][fail_attr] = value
    return goal


class MyEncoder(json.JSONEncoder):

    def default(self, obj):
        if type(obj) == 'bool_':
            return bool(obj)
        return super(MyEncoder, self).default(obj)


def main():
    print('1')
    d, dp, id, idp, fid, fidp, rd, rdp, bd, bdp, bv, bvp, fbd, fbdp, fp, rp, bp, bfp, t, tp = preprocessing(
        'soloist-main/examples/multiwoz/data/multi-woz/delex.json')
    print('2')
    db, dbc = get_database('soloist-main/examples/multiwoz/db/')
    print('3')
    goal_list = []
    for i in tqdm(range(1000), total=1000):
        goal = generate_goal(d, dp, id, idp, fid, fidp, rd, rdp, bd, bdp, bv, bvp, fbd, fbdp, fp, rp, bp, bfp, t, tp,
                             db, dbc, real=False)
        goal_list.append(goal)
    print('4')
    with open('dataset/mwoz/fake_goals.json', 'w') as f:
        json.dump(goal_list, f, cls=MyEncoder, indent=4)


if __name__ == '__main__':
    main()
