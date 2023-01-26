from random import shuffle,randint,choice
import sys
import pickle
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def next_batch_pairwise(data,batch_size,n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)

    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])

            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])
        yield u_idx, i_idx, j_idx


def mixup_next_batch_pairwise(data,batch_size,n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)

    import pickle
    popular_item_list = pickle.load(open("{}/popular_item_list_int.pkl".format('./dataset/ml-1M'), "rb"))

    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])

        dict_ = {}
        o_idx, p1_idx,p2_idx, q1_idx, q2_idx = [],[],[],[],[]

        for i in range(len(u_idx)):

            val = dict_.get(u_idx[i], -1)

            if val == -1:
                dict_[u_idx[i]] = i_idx[i]
                dict_[-u_idx[i]] = j_idx[i]
            else:
                dict_[u_idx[i]] = -1
                dict_[-u_idx[i]] = -1
                if val not in popular_item_list and i_idx[i] in popular_item_list:
                    o_idx.append(u_idx[i])
                    p1_idx.append(val)
                    q1_idx.append(dict_[-u_idx[i]])
                    p2_idx.append(i_idx[i])
                    q2_idx.append(j_idx[i])
                if val in popular_item_list and i_idx[i] not in popular_item_list:
                    o_idx.append(u_idx[i])
                    p2_idx.append(val)
                    q2_idx.append(dict_[-u_idx[i]])
                    p1_idx.append(i_idx[i])
                    q1_idx.append(j_idx[i])

        yield u_idx, i_idx, j_idx,o_idx, p1_idx,p2_idx, q1_idx, q2_idx


def synthesize_next_batch_pairwise(data,batch_size,n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)
    import pickle
    popular_item_list = pickle.load(open("{}/popular_item_list.pkl".format('./dataset/ml-1M'), "rb"))
    popular_item_list = [str(_) for _ in popular_item_list]

    item_sim_dict = pickle.load(open("{}/item_index_rank_dict.pkl".format('./dataset/ml-1M'), "rb"))
    inverse_item_sim_dict = {}
    for i in item_sim_dict:
        inv_i = item_sim_dict[i][1]
        temp = inverse_item_sim_dict.get(inv_i,[])
        temp.append(i)
        inverse_item_sim_dict[inv_i] = temp
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []

        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])

            if items[i] in popular_item_list:
                if items[i] in inverse_item_sim_dict:
                    replace_tail_item = (choice(inverse_item_sim_dict[(items[i])]))
                    i_idx.append(data.item[replace_tail_item])
                    u_idx.append(data.user[user])
                    j_idx.append(data.item[neg_item])

        yield u_idx, i_idx, j_idx


def adaptive_next_batch_pairwise(data,batch_size,n_negs=1,popular_item_list_str=[]):
    training_data = data.training_data
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)

    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        lt_idx = []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])

            lt_idx.append(items[i] in popular_item_list_str)

            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])
        yield u_idx, i_idx, j_idx,lt_idx


def selective_next_batch_pairwise(data,batch_size,potential_list,item_a,item_b, n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)

    item_a_ = [_ for _ in range(len(item_a))]
    print('add edge number',len(item_a))
    data.item_inv = {v: k for k, v in data.item.items()}

    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        lt_idx = []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            lt_idx.append(potential_list.get(items[i],0) )
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])
        import random
        item_a_selected = random.sample(item_a_, 1)
        item_a2=[item_a[i] for i in item_a_selected]

        item_b2=[item_b[i] for i in item_a_selected]
        
        user_pos = []
        item_neg = []

        for i in range(len(item_a2)):
            pos_user = choice(list(data.training_set_i[data.item_inv[item_b2[i]]].keys()))
            neg_item = choice(item_list)
            while neg_item in data.training_set_u[pos_user] or neg_item==data.item_inv[item_a2[i]] or neg_item==data.item_inv[item_b2[i]]:
                neg_item = choice(item_list)

            user_pos.append(data.user[pos_user])
            item_neg.append(data.item[neg_item])
            lt_idx.append( 0.5 )

        yield u_idx+user_pos, i_idx, j_idx+item_neg,lt_idx,item_a2,item_b2

def galr_next_batch_pairwise(data,batch_size,potential_list, n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)

    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        lt_idx = []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])

            lt_idx.append(potential_list.get(items[i],0) )

            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])

        yield u_idx, i_idx, j_idx,lt_idx


import numpy as np
def adaptive_dropout_next_batch_pairwise(data,batch_size,sample_idx,n_negs=1):

    training_data = list(np.array(data.training_data)[sample_idx])
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)

    import pickle
    popupar_item_list = pickle.load(open("{}/popupar_item_list.pkl".format('./dataset/ml-1M'), "rb"))
    popupar_item_list = [str(_) for _ in popupar_item_list]

    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        lt_idx = []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])

            lt_idx.append(items[i] in popupar_item_list)

            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])
        yield u_idx, i_idx, j_idx,lt_idx



def sim_next_batch_pairwise(data,batch_size,n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)

    import pickle
    popupar_item_list = pickle.load(open("{}/popupar_item_list.pkl".format('./dataset/ml-1M'), "rb"))
    popupar_item_list = [str(_) for _ in popupar_item_list]
    item_sim_dict = pickle.load(open("{}/item_sim_list.pkl".format('./fig'), "rb"))
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        lt_idx = []
        lt2_idx = []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            if True:
                i_idx.append(data.item[items[i]])
                u_idx.append(data.user[user])
                import random
                lt_idx.append( random.sample(item_sim_dict[data.item[items[i]]][:3],1) )
                lt2_idx.append( random.sample(item_sim_dict[data.item[items[i]]][350:],1) )

                for m in range(n_negs):
                    neg_item = choice(item_list)
                    while neg_item in data.training_set_u[user]:
                        neg_item = choice(item_list)
                    j_idx.append(data.item[neg_item])
        yield u_idx, i_idx, j_idx,lt_idx,lt2_idx


def next_batch_pointwise(data,batch_size):
    training_data = data.training_data
    data_size = len(training_data)
    batch_id = 0
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, y = [], [], []
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            y.append(1)
            for instance in range(4):
                item_j = randint(0, data.item_num - 1)
                while data.id2item[item_j] in data.training_set_u[user]:
                    item_j = randint(0, data.item_num - 1)
                u_idx.append(data.user[user])
                i_idx.append(item_j)
                y.append(0)
        yield u_idx, i_idx, y