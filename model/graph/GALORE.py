import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import *
from base.torch_interface import TorchGraphInterface
from util.loss_torch import *
from data.augmentor import GraphAugmentor
import pickle
import sys
import scipy.sparse as sp
torch.cuda.current_device()


class GALORE(GraphRecommender):
    def __init__(self, conf, training_set, test_set,test_set1,test_set2,valid_set):
        super(GALORE, self).__init__(conf, training_set, test_set,test_set1,test_set2,valid_set)
        args = OptionConf(self.config['GALORE'])
        self.n_layers = int(args['-n_layer'])
        self.train_mode = self.config['train_mode']
        self.emb_size_ = self.config['embbedding.size']
        self.cluster_num = float(self.config['cluster_num'])
        self.cl_rate = 0.5
        self.eps = 0.1
        self.temp = 0.15

        self.n_layers = int(args['-n_layer'])
        self.layer_cl = int(args['-l*'])
        self.model = GALORE_Encoder(self.data, self.emb_size, self.eps, self.n_layers,self.layer_cl)
        self.drop_rate = float(self.config['drop_rate'])
        self.dataset_dir = f"./dataset/{self.config['dataset']}"


    def train(self):
        model = self.model.cuda()
        save_model_dir = f'./model_cpt/{self.config["dataset"]}'
        ### load a pre-trained recommendation model ###
        model.load_state_dict(torch.load(save_model_dir + f'/lightgcn_best_{self.emb_size_}.pt'))
        self.user_emb, self.item_emb = model._get_embedding()
        model._load_model(self.user_emb,self.item_emb)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        self.maxEpoch = 60
        popupar_item_list = pickle.load(open("{}/popular_item_list.pkl".format(self.dataset_dir), "rb"))
        popular_item_list_str = [str(_) for _ in popupar_item_list]
        
        ### Drop edge ###
        user_lc_dict=pickle.load(open("{}/user_lc_dict.pkl".format(self.dataset_dir), "rb")) 
        item_lc_dict=pickle.load(open("{}/item_lc_dict.pkl".format(self.dataset_dir), "rb")) 
        user_lc_dict_ = {}
        item_lc_dict_= {}
        popular_item_list_str =pickle.load(open("{}/popular_item_list.pkl".format(self.dataset_dir), "rb")) 
        popular_item_list= []

        for i in user_lc_dict:
            if i in self.data.user:
                user_lc_dict_[self.data.user[i]]=user_lc_dict[i]
        for i in item_lc_dict:
            if i in self.data.item:
                item_lc_dict_[self.data.item[i]]=item_lc_dict[i]

        for i in popular_item_list_str:
            if i in self.data.item:
                popular_item_list.append(self.data.item[i])

        ### Add edge ###
        SIM_FILE_PATH = "{}/sim_dict_galr.pkl".format(self.dataset_dir)
        if not os.path.exists("{}/sim_dict_galr.pkl".format(self.dataset_dir)):
            a = self.data.interaction_mat.T * self.data.interaction_mat 
            a = a.toarray()
            sim_dict = {}
            for i in range(a.shape[0]):
                if not i in popular_item_list:
                    sorted_arr = sorted(range(len(a[i])), key=lambda k: a[i][k])[::-1]
                    sim_dict[i] = np.array(sorted_arr)[np.in1d(np.array(sorted_arr), np.array(popular_item_list))]
                    
            pickle.dump(sim_dict,open(SIM_FILE_PATH, "wb")) 
        else:
            sim_dict = pickle.load(open(SIM_FILE_PATH, "rb"))


        for epoch in range(self.maxEpoch):
            local_nor_adj,_,_ = model.graph_add_edge(sim_dict,user_lc_dict_, item_lc_dict_,drop_rate=self.drop_rate)
            for n, batch in enumerate(adaptive_next_batch_pairwise(self.data, self.batch_size, popular_item_list_str=popular_item_list_str)):
                user_idx, pos_idx, neg_idx, lt_idx = batch
                rec_user_emb, rec_item_emb = model(perturbed_adj=local_nor_adj,perturbed=False)#sparse_norm_adj=local_nor_adj
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                # Sampling Mixup #
                random_v = ( torch.rand( (user_emb.shape[0],1) ).cuda() * 0.5 ).repeat(1,user_emb.shape[1])
                neg_item_emb = (1 - random_v) * neg_item_emb + random_v * pos_item_emb  
                batch_loss =  bpr_loss(user_emb, pos_item_emb, neg_item_emb)+ l2_reg_loss(self.reg, user_emb,pos_item_emb)
                # adaptive_bpr_loss_upsampling(user_emb, pos_item_emb, neg_item_emb,lt_idx,upindex=self.cluster_num) 
                # neg_item_emb)/self.batch_size
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            if epoch> 5 and epoch % 2 == 0:
                self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb


    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()


class GALORE_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, layer_cl):
        super(GALORE_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.layer_cl = layer_cl
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

        self.aug_type = 1
        self.drop_rate=0.1
        self.temp = 0.15

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def _load_model(self,user_emb, item_emb):

        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(user_emb),
            'item_emb': nn.Parameter(item_emb)
        })
        return 
    def _get_embedding(self):
        return self.embedding_dict.user_emb, self.embedding_dict.item_emb

    def graph_reconstruction(self):
        if self.aug_type==0 or 1:
            dropped_adj = self.random_graph_augment()
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj

    def random_graph_augment(self):
        dropped_mat = None
        if self.aug_type == 0:
            dropped_mat = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
        elif self.aug_type == 1 or self.aug_type == 2:
            dropped_mat = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
        elif self.aug_type == 4:
            dropped_mat = GraphAugmentor.adaptive_edge_dropout(self.data.interaction_mat, 0.15, 0.5)
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()


    def graph_add_edge(self,sim_dict,user_lc_dict,item_lc_dict,drop_rate, edge_num = 2, n_cluster = 3):
        sp_adj = self.data.interaction_mat
        adj_shape = sp_adj.get_shape()
        edge_count = sp_adj.count_nonzero()
        row_idx, col_idx = sp_adj.nonzero()
        # keep_idx = random.sample(range(edge_count), int(edge_count * (1 - drop_rate)))
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)


        '''
        adaptive dropedge
        '''
        proba_list = np.ones_like(user_np, dtype=np.float32)
        user_np_length = user_np.shape[0]
        for i in range(user_np_length):
            proba_list[i] = user_lc_dict[user_np[i]] * item_lc_dict[item_np[i]]
           
        from numpy.random import choice
        proba_list /= proba_list.sum()
        keep_idx = choice([_ for _ in range(user_np_length)], int(user_np_length * (1-drop_rate)),
                    p=proba_list)
        user_np = user_np[keep_idx]
        item_np = item_np[keep_idx]

        item_sim_dict = sim_dict

        # clustering 
        item_emb = self.embedding_dict['item_emb'].cpu().detach().numpy()
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(item_emb)
        clu_result = list(kmeans.labels_)

        item_a=[]
        item_b=[]

        user_np = user_np

        for i in item_sim_dict:
            # if i in self.data.item:
            import random
            ran_v = random.randint(0, edge_num)
            rdm_itm_ = item_sim_dict[i][ran_v]
            # for rdm_itm_ in rdm_itm:
                # if rdm_itm_ in self.data.item:
            if clu_result[ i ]==clu_result[ rdm_itm_ ]:
                # print(user_np.shape)
                item_a.append( i )
                item_b.append( rdm_itm_ )

        item_a_ = [_  + self.data.user_num for _ in item_a]
        print('add edge num', len(item_a_))
        user_np = np.append(user_np, item_a_)
        item_np = np.append(item_np, item_b)
        item_np = item_np+self.data.user_num
        edges = np.ones_like(user_np, dtype=np.float32)
        n_nodes = self.data.user_num + self.data.item_num
        dropped_adj = sp.csr_matrix((edges, (user_np, item_np)), shape=(n_nodes, n_nodes), dtype=np.float32)
        dropped_mat = self.data.convert_to_laplacian_mat_galr(dropped_adj)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda(), item_a,item_b


    def forward(self,perturbed_adj=None,perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        all_embeddings_cl = ego_embeddings
        for k in range(self.n_layers):
            # ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed_adj is not None:
                if isinstance(perturbed_adj,list):
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)

            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            
            
            all_embeddings.append(ego_embeddings)
            if k==self.layer_cl-1:
                all_embeddings_cl = ego_embeddings
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.data.user_num, self.data.item_num])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embeddings_cl, [self.data.user_num, self.data.item_num])
        if perturbed:
            return user_all_embeddings, item_all_embeddings,user_all_embeddings_cl, item_all_embeddings_cl
        return user_all_embeddings,  item_all_embeddings
