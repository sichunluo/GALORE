import numpy as np
import random
import scipy.sparse as sp
import sys

class GraphAugmentor(object):
    def __init__(self):
        pass

    @staticmethod
    def node_dropout(sp_adj, drop_rate):
        """Input: a sparse adjacency matrix and a dropout rate."""
        adj_shape = sp_adj.get_shape()
        row_idx, col_idx = sp_adj.nonzero()
        drop_user_idx = random.sample(range(adj_shape[0]), int(adj_shape[0] * drop_rate))
        drop_item_idx = random.sample(range(adj_shape[1]), int(adj_shape[1] * drop_rate))
        indicator_user = np.ones(adj_shape[0], dtype=np.float32)
        indicator_item = np.ones(adj_shape[1], dtype=np.float32)
        indicator_user[drop_user_idx] = 0.
        indicator_item[drop_item_idx] = 0.
        diag_indicator_user = sp.diags(indicator_user)
        diag_indicator_item = sp.diags(indicator_item)
        mat = sp.csr_matrix(
            (np.ones_like(row_idx, dtype=np.float32), (row_idx, col_idx)),
            shape=(adj_shape[0], adj_shape[1]))
        mat_prime = diag_indicator_user.dot(mat).dot(diag_indicator_item)
        return mat_prime

    @staticmethod
    def edge_dropout(sp_adj, drop_rate):
        """Input: a sparse user-item adjacency matrix and a dropout rate."""
        adj_shape = sp_adj.get_shape()
        edge_count = sp_adj.count_nonzero()
        row_idx, col_idx = sp_adj.nonzero()
        keep_idx = random.sample(range(edge_count), int(edge_count * (1 - drop_rate)))
        user_np = np.array(row_idx)[keep_idx]
        item_np = np.array(col_idx)[keep_idx]
        edges = np.ones_like(user_np, dtype=np.float32)
        dropped_adj = sp.csr_matrix((edges, (user_np, item_np)), shape=adj_shape)
        return dropped_adj


    @staticmethod
    def edge_dropout_GALR(sp_adj, drop_rate, user_lc_dict, item_lc_dict):
        """Input: a sparse user-item adjacency matrix and a dropout rate."""
        adj_shape = sp_adj.get_shape()
        edge_count = sp_adj.count_nonzero()
        row_idx, col_idx = sp_adj.nonzero()
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        # print(user_np.max(), user_np.min(), item_np.max(), item_np.min())
        # item_np = item_np-user_num
        proba_list = np.ones_like(user_np, dtype=np.float32)
        user_np_length = user_np.shape[0]
        # print(edge_count, user_np_length)
        for i in range(user_np_length):
            proba_list[i] = user_lc_dict[user_np[i]] * item_lc_dict[item_np[i]]
        
        from numpy.random import choice
        proba_list /= proba_list.sum()
        keep_idx = choice([_ for _ in range(user_np_length)], int(user_np_length * (1-drop_rate)),
                    p=proba_list)
        user_np = user_np[keep_idx]
        item_np = item_np[keep_idx]

        edges = np.ones_like(user_np, dtype=np.float32)
        dropped_adj = sp.csr_matrix((edges, (user_np, item_np)), shape=adj_shape)
        return dropped_adj


    @staticmethod
    def adaptive_edge_dropout(sp_adj, drop_rate1, drop_rate2):
        """Input: a sparse user-item adjacency matrix and a dropout rate."""

        adj_shape = sp_adj.get_shape()
        edge_count = sp_adj.count_nonzero()
        row_idx, col_idx = sp_adj.nonzero()
        import os.path
        import pickle
        from tqdm import tqdm

        if not os.path.exists('./dataset/ml-1M/head_list.pkl'):

            popular_item_list = pickle.load(open("{}/popular_item_list_int.pkl".format('./dataset/ml-1M'), "rb"))
            head_list = []
            tail_list = []
            for i in tqdm(range(len(col_idx))):
                if col_idx[i] in popular_item_list:
                    head_list.append(i)
                else:
                    tail_list.append(i)
            pickle.dump(head_list,open("{}/head_list.pkl".format('./dataset/ml-1M'), "wb"))
            pickle.dump(tail_list,open("{}/tail_list.pkl".format('./dataset/ml-1M'), "wb"))
            print('Get /head_list.pkl..')
        else:
            head_list =pickle.load(open("{}/head_list.pkl".format('./dataset/ml-1M'), "rb"))
            tail_list =pickle.load(open("{}/tail_list.pkl".format('./dataset/ml-1M'), "rb"))

        keep_idx1 = random.sample( head_list , int(len(head_list) * (1 - drop_rate1)))
        keep_idx2 = random.sample( tail_list , int(len(tail_list) * (1 - drop_rate2)))
        keep_idx = keep_idx1 + keep_idx2

        user_np = np.array(row_idx)[keep_idx]
        item_np = np.array(col_idx)[keep_idx]
        edges = np.ones_like(user_np, dtype=np.float32)
        dropped_adj = sp.csr_matrix((edges, (user_np, item_np)), shape=adj_shape)
        return dropped_adj

