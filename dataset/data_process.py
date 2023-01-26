import pandas as pd
from tqdm import tqdm
import torch
# from torch_ppr import personalized_page_rank
import numpy as np
# from torch_ppr import page_rank
import pickle
import random

def index_rank():
    df = pd.read_csv('train.txt',header=None,names=['user','item','rating'],sep=' ')
    item_max_num = df['item'].max() + 1
    user_item_dict = {}
    for idx, row in tqdm(df.iterrows()):
        sav_var = user_item_dict.get(str(row['item']), list())
        sav_var.append(row['user'])
        user_item_dict[str(row['item'])] = sav_var

    df['user']=df['user'].apply(lambda x: x+item_max_num)
    subset = df[['item', 'user']]
    tuples = [tuple(x) for x in subset.to_numpy()]

    edge_index = torch.as_tensor(data= tuples ).t()
    result = personalized_page_rank(edge_index=edge_index, indices=[_ for _ in range(0, item_max_num)]).cpu().numpy()
    print(result,result.shape)
    index_rank_dict={}
    popular_item_index_rank_dict={}
    item_index_rank_dict={}
    popular_item_list = pickle.load(open("{}/popular_item_list.pkl".format('.'), "rb"))
    popular_item_list_int = [int(n) for n in popular_item_list]
    user_index_rank_dict={}
    for i in tqdm(range(len(result))):
                i = str(i)
                if i not in popular_item_list:              
                    index_rank = np.argsort(result[int(i)][:item_max_num])[::-1]
                    popular_item_index_rank = list(index_rank[np.in1d(np.array(index_rank), np.array(popular_item_list_int))])
                    item_index_rank_dict[i] = [str(n) for n in index_rank]
                    popular_item_index_rank_dict[i] = [str(n) for n in popular_item_index_rank]
                    selected_user = result[int(i)][item_max_num:]
                    if i in user_item_dict:
                        for _ in user_item_dict[i]:
                            selected_user[_] = -1000

                    index_rank = np.argsort(selected_user)[::-1]
                    user_index_rank_dict[i] = [str(n) for n in index_rank]


    pickle.dump(popular_item_index_rank_dict, open("{}/popular_item_index_rank_dict.pkl".format('.'), "wb"))
    pickle.dump(item_index_rank_dict, open("{}/item_index_rank_dict.pkl".format('.'), "wb"))
    pickle.dump(user_index_rank_dict, open("{}/user_index_rank_dict.pkl".format('.'), "wb"))

    print(popular_item_index_rank_dict['1'][:10],item_index_rank_dict['1'][:10],user_index_rank_dict['1'][:10],)
    print(len(popular_item_index_rank_dict.keys()),len(user_index_rank_dict.keys()),len(item_index_rank_dict.keys()),)


def testset_split():
    train_data = pd.read_csv('train_raw.txt', header=None, sep=' ',names=['user', 'item'], usecols=[0, 1])
    whole_item_list = train_data['item'].value_counts()
    print(whole_item_list)
    item_num = train_data['item'].nunique()
    popupar_item_list = whole_item_list[ : int(0.2 * item_num)]
    popupar_item_list=list(popupar_item_list.index.values)

    test_data = pd.read_csv('test.txt', header=None, sep=' ',names=['user', 'item','rating'], usecols=[0, 1,2])
    test_data_head = test_data.loc[test_data['item'].isin(popupar_item_list)]
    print(test_data_head)
    test_data_tail = test_data.loc[~test_data['item'].isin(popupar_item_list)]
    print(test_data_tail)
    test_data_head.to_csv('test1.txt', header=None, index=None, sep=' ')
    test_data_tail.to_csv('test2.txt', header=None, index=None, sep=' ')

    popular_item_list= [str(_) for _ in popupar_item_list]
    pickle.dump(popular_item_list, open("{}/popular_item_list.pkl".format('.'), "wb"))
    print(len(popular_item_list), popular_item_list[:10])

def valid_set_split():
    train = []
    test = []
    with open('train_raw.txt') as f:
        for line in f:
            new_line = line
            if random.random() < 0.125:
                test.append(new_line)
            else:
                train.append(new_line)

    with open('train.txt','w') as f:
        f.writelines(train)

    with open('valid.txt','w') as f:
        f.writelines(test)


def get_sim_dict():
    import math

    train_data = pd.read_csv('train_raw.txt', header=None, sep=' ',names=['user', 'item'], usecols=[0, 1])
    whole_item_list = train_data['item'].value_counts().to_frame()
    whole_item_list['item_name'] = whole_item_list.index
    whole_item_list['item'] = whole_item_list['item'].apply(lambda x: pow(math.log(x+1),-1))
    whole_item_list['item'] = whole_item_list['item'].apply(lambda x: min(x,1))
    num_min = whole_item_list['item'].min()
    num_max = whole_item_list['item'].max()
    whole_item_list['item'] = whole_item_list['item'].apply(lambda x: max( (num_max-x)/(num_max-num_min) ,0.5))
    whole_item_list['item'] = whole_item_list['item'].apply(lambda x: min(x,1))
    item_number = whole_item_list["item"].values.tolist()
    item_name = whole_item_list["item_name"].values.tolist()

    item_dict = {}
    for i in range(len(item_name)):
        item_dict[str(item_name[i])]=item_number[i]

    pickle.dump(item_dict, open("{}/item_lc_dict.pkl".format('.'), "wb"))

    whole_item_list = train_data['user'].value_counts().to_frame()
    whole_item_list = whole_item_list.rename({'user': 'item'},axis='columns')

    whole_item_list['item_name'] = whole_item_list.index
    whole_item_list['item'] = whole_item_list['item'].apply(lambda x: pow(math.log(x+1),-1))
    whole_item_list['item'] = whole_item_list['item'].apply(lambda x: min(x,1))
    num_min = whole_item_list['item'].min()
    num_max = whole_item_list['item'].max()
    whole_item_list['item'] = whole_item_list['item'].apply(lambda x: max( (x-num_min)/(num_max-num_min) ,0.5))
    # print(whole_item_list)


    whole_item_list['item'] = whole_item_list['item'].apply(lambda x: min(x,1))
    item_number = whole_item_list["item"].values.tolist()
    item_name = whole_item_list["item_name"].values.tolist()

    item_dict = {}
    for i in range(len(item_name)):
        item_dict[str(item_name[i])]=item_number[i]

    pickle.dump(item_dict, open("{}/user_lc_dict.pkl".format('.'), "wb"))




if __name__ == '__main__':
    valid_set_split()
    testset_split()
    # index_rank()
    get_sim_dict()

    




    
    


    
    



