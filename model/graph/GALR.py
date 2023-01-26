# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from base.graph_recommender import GraphRecommender
# from util.conf import OptionConf
# from util.sampler import next_batch_pairwise,adaptive_next_batch_pairwise
# from base.torch_interface import TorchGraphInterface
# from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE,ItemInfoNCE
# import sys
# # Paper: XSimGCL - Towards Extremely Simple Graph Contrastive Learning for Recommendation


# class XSimGCL(GraphRecommender):
#     def __init__(self, conf, training_set, test_set,test_set1,test_set2,valid_set):
#         super(XSimGCL, self).__init__(conf, training_set, test_set,test_set1,test_set2,valid_set)
#         args = OptionConf(self.config['XSimGCL'])
#         self.cl_rate = float(args['-lambda'])
#         self.eps = float(args['-eps'])
#         self.temp = float(args['-tau'])
#         self.n_layers = int(args['-n_layer'])
#         self.layer_cl = int(args['-l*'])
#         self.model = XSimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers,self.layer_cl)

#     def train(self):
#         model = self.model.cuda()
        
#         optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
#         # model = torch.load_state_dict()
#         # model.load_state_dict(torch.load('./lightgcn_model.pt'))

#         # model = self.model.cuda()
#         # model.load_state_dict(torch.load('./xsimgcl_model.pt'))
#         # self.user_emb, self.item_emb = model._get_embedding()
#         # model._load_model(self.user_emb,self.item_emb)
#         # optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
#         # self.draw_item_performance_figure(file_name='XSimGCL22')
        

#         self.maxEpoch = 100

#         for epoch in range(self.maxEpoch):
#             for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
#                 user_idx, pos_idx, neg_idx = batch
#                 rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb  = model(True)
#                 user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
#                 rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
#                 # 

#                 # 重写cl_loss
#                 # print(lt_idx,pos_idx)
#                 # print(len(lt_idx),len(pos_idx))
#                 # print(lt_idx.shape,pos_idx.shape)
#                 # if epoch<7:
#                 cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx],rec_user_emb,cl_user_emb,rec_item_emb,cl_item_emb)
#                 # else:
#                 # cl_loss2 =self.cl_rate * self.my_cal_cl_loss2([lt_idx,lt2_idx,pos_idx],rec_item_emb,cl_item_emb)  

#                 batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss

                
#                 # print(user_idx) + cl_loss2
#                 # print(rec_loss)
#                 # import sys
#                 # sys.exit(1)
#                 # Backward and optimize
#                 optimizer.zero_grad()
#                 batch_loss.backward()
#                 optimizer.step()
#                 if n % 100==0:
#                     print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss',cl_loss.item()) # 
#             with torch.no_grad():
#                 self.user_emb, self.item_emb = self.model()
#             self.fast_evaluation(epoch)

#         # self.draw_item_performance_figure(file_name='XSimGCL3')



        



#         # fine tune !
#         # for epoch in range(self.maxEpoch):
#         #     for n, batch in enumerate(my_next_batch_pairwise(self.data, self.batch_size)):
#         #         user_idx, pos_idx, neg_idx,lt_idx,lt2_idx = batch
#         #         rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb  = model(True)
#         #         user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
#         #         rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
#         #         # 

#         #         # 重写cl_loss
#         #         # print(lt_idx,pos_idx)
#         #         # print(len(lt_idx),len(pos_idx))
#         #         # print(lt_idx.shape,pos_idx.shape)
#         #         # if epoch<7:
#         #         # cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx],rec_user_emb,cl_user_emb,rec_item_emb,cl_item_emb)
#         #         # else:
#         #         #     
#         #         self.cl_rate = 100000000
#         #         cl_loss = self.cl_rate * self.my_cal_cl_loss([lt_idx,lt2_idx,pos_idx],rec_item_emb) 

#         #         batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss

                
#         #         # print(user_idx)
#         #         # print(rec_loss)
#         #         # import sys
#         #         # sys.exit(1)
#         #         # Backward and optimize
#         #         optimizer.zero_grad()
#         #         batch_loss.backward()
#         #         optimizer.step()
#         #         if n % 100==0:
#         #             print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss',cl_loss.item()) # 
#         #     with torch.no_grad():
#         #         self.user_emb, self.item_emb = self.model()
#         #     self.fast_evaluation(epoch)
        

#         self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

#         # torch.save(model.state_dict(), './xsimgcl_model.pt')
        


#     def cal_cl_loss2(self, idx, user_view1,user_view2,item_view1,item_view2,lt_idx):
#         u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
#         i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
#         user_cl_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)

#         def get_first(A):
#             unique, idx, counts = torch.unique(A,  sorted=True, return_inverse=True, return_counts=True)
#             _, ind_sorted = torch.sort(idx)
#             cum_sum = counts.cumsum(0)
#             cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
#             first_indicies = ind_sorted[cum_sum]
#             return first_indicies

#         i_idx, inverse_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long), return_inverse=True)
#         first_indicies = get_first(torch.Tensor(idx[1]).type(torch.long))
#         # inverse_idx_ = torch.unique(inverse_idx)
#         # inverse_idx_ = torch.Tensor(inverse_idx_).type(torch.long)
#         # print(inverse_idx_, type(inverse_idx_))
#         # print(inverse_idx_.numpy(), type(inverse_idx_.numpy()))
#         # print(lt_idx, type(lt_idx))
#         # print(first_indicies)
#         lt_idx_ = torch.tensor(lt_idx, dtype=torch.long)
#         lt_idx_ = lt_idx_[first_indicies]
#         # sys.exit(1)

#         item_cl_loss = ItemInfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp,lt_idx_)


        
#         # .cuda()

#         # print(item_cl_loss, item_cl_loss.shape)
#         # sys.exit(1)

#         return user_cl_loss + item_cl_loss

#     def cal_cl_loss(self, idx, user_view1,user_view2,item_view1,item_view2):
#         u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
#         i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
#         user_cl_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
#         item_cl_loss = InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)


        
#         # .cuda()

#         # print(item_cl_loss, item_cl_loss.shape)
#         # sys.exit(1)

#         return user_cl_loss + item_cl_loss

#     def cal_cl_loss_user(self, idx, user_view1,user_view2,item_view1,item_view2):
#         u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
#         i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()

#         # print(u_idx.shape, i_idx.shape,i_idx)
#         user_cl_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
#         item_cl_loss = InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)
#         return user_cl_loss 

#     def my_cal_cl_loss(self, idx, item_view1):
#         i_idx1 = (torch.Tensor(idx[0]).type(torch.long)).cuda()
#         i_idx2 = (torch.Tensor(idx[1]).type(torch.long)).cuda()
#         i_idx3 = (torch.Tensor(idx[2]).type(torch.long)).cuda()

        
#         # user_cl_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
#         item_cl_loss = InfoNCE(item_view1[i_idx1], item_view1[i_idx3], self.temp)
#         item_cl_loss2 = InfoNCE(item_view1[i_idx2], item_view1[i_idx3], self.temp)
#         return  item_cl_loss

#     def my_cal_cl_loss2(self, idx, item_view1, item_view2):
#         i_idx1 = (torch.Tensor(idx[0]).type(torch.long)).squeeze().cuda()
#         i_idx2 = (torch.Tensor(idx[1]).type(torch.long)).cuda()
#         i_idx3 = (torch.Tensor(idx[2]).type(torch.long)).cuda()

#         # print(i_idx1.shape, i_idx3.shape,i_idx1)
#         # import sys
#         # sys.exit(1)
#         # user_cl_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
#         item_cl_loss = InfoNCE(item_view1[i_idx1], item_view1[i_idx3], self.temp)
#         item_cl_loss2 = InfoNCE(item_view1[i_idx2], item_view1[i_idx3], self.temp)
#         return  item_cl_loss


#     def save(self):
#         with torch.no_grad():
#             self.best_user_emb, self.best_item_emb = self.model.forward()

#     def predict(self, u):
#         with torch.no_grad():
#             u = self.data.get_user_id(u)
#             score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
#             return score.cpu().numpy()


# class XSimGCL_Encoder(nn.Module):
#     def __init__(self, data, emb_size, eps, n_layers, layer_cl):
#         super(XSimGCL_Encoder, self).__init__()
#         self.data = data
#         self.eps = eps
#         self.emb_size = emb_size
#         self.n_layers = n_layers
#         self.layer_cl = layer_cl
#         self.norm_adj = data.norm_adj
#         self.embedding_dict = self._init_model()
#         self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

#     def _init_model(self):
#         initializer = nn.init.xavier_uniform_
#         embedding_dict = nn.ParameterDict({
#             'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
#             'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
#         })
#         return embedding_dict
    
#     def _get_embedding(self):
#         return self.embedding_dict.user_emb, self.embedding_dict.item_emb

#     def _load_model(self,user_emb, item_emb):
#         self.embedding_dict = nn.ParameterDict({
#             'user_emb': nn.Parameter(user_emb),
#             'item_emb': nn.Parameter(item_emb)
#         })
#         return 

#     def forward(self, perturbed=False):
#         ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
#         all_embeddings = []
#         all_embeddings_cl = ego_embeddings
#         for k in range(self.n_layers):
#             ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
#             if perturbed:
#                 random_noise = torch.rand_like(ego_embeddings).cuda()
#                 ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
#             all_embeddings.append(ego_embeddings)
#             if k==self.layer_cl-1:
#                 all_embeddings_cl = ego_embeddings
#         final_embeddings = torch.stack(all_embeddings, dim=1)
#         final_embeddings = torch.mean(final_embeddings, dim=1)
#         user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.data.user_num, self.data.item_num])
#         user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embeddings_cl, [self.data.user_num, self.data.item_num])
#         if perturbed:
#             return user_all_embeddings, item_all_embeddings,user_all_embeddings_cl, item_all_embeddings_cl
#         return user_all_embeddings, item_all_embeddings

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
# Paper: XSimGCL - Towards Extremely Simple Graph Contrastive Learning for Recommendation


class GALR(GraphRecommender):
    def __init__(self, conf, training_set, test_set,test_set1,test_set2,valid_set):
        super(GALR, self).__init__(conf, training_set, test_set,test_set1,test_set2,valid_set)
        args = OptionConf(self.config['GALR'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.temp = float(args['-tau'])
        self.n_layers = int(args['-n_layer'])
        self.layer_cl = int(args['-l*'])
        self.model = GALR_Encoder(self.data, self.emb_size, self.eps, self.n_layers,self.layer_cl)

        self.train_mode = self.config['train_mode']
        self.dataset_dir = f"./dataset/{self.config['dataset']}"

        self.drop_rate = 0.1


    
    def train_normal(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        print('dajnisdnai')
    
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

        print('weqsdasda')

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
            # dropped_adj1 = model.graph_reconstruction()
            # dropped_adj2 = model.graph_reconstruction()

            print('fgafadasdad')

            dropped_adj1=GraphAugmentor.edge_dropout_GALR(self.data.interaction_mat, self.drop_rate, user_lc_dict_, item_lc_dict_)
            dropped_adj1 = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.convert_to_laplacian_mat(dropped_adj1)).cuda()
            
            dropped_adj2=GraphAugmentor.edge_dropout_GALR(self.data.interaction_mat, self.drop_rate, user_lc_dict_, item_lc_dict_)
            dropped_adj2 = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.convert_to_laplacian_mat(dropped_adj2)).cuda()


            print('kokookok')

            perturbed_mat2,_,_ = model.graph_add_edge(sim_dict)

            print('pkponionon ijn')
            
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb  = model(perturbed_adj=perturbed_mat2,perturbed=False)
                # rec_user_emb2, rec_item_emb2, cl_user_emb2, cl_item_emb2  = model(perturbed=True)
                cl_loss_sgl = model.cal_cl_loss_sgl([user_idx,pos_idx],dropped_adj1,dropped_adj2)
                cl_loss_sgl = self.cl_rate *cl_loss_sgl

                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                # cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx],rec_user_emb,cl_user_emb,rec_item_emb,cl_item_emb) * 0.5

                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb)  + cl_loss_sgl 
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss_sgl.item(),)
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

        model._load_model(self.user_emb,self.item_emb)
        torch.save(model.state_dict(), f"./model_cpt/galr_{self.config['dataset']}.pt")

    
    def train_reweighting(self):
        self.model = GALR_Encoder(self.data, self.emb_size, self.eps, self.n_layers,self.layer_cl)
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        self.msg = f'GALR'
    
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
            # dropped_adj1 = model.graph_reconstruction()
            # dropped_adj2 = model.graph_reconstruction()
            dropped_adj1=GraphAugmentor.edge_dropout_GALR(self.data.interaction_mat, self.drop_rate, user_lc_dict_, item_lc_dict_)
            dropped_adj1 = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.convert_to_laplacian_mat(dropped_adj1)).cuda()
            
            dropped_adj2=GraphAugmentor.edge_dropout_GALR(self.data.interaction_mat, self.drop_rate, user_lc_dict_, item_lc_dict_)
            dropped_adj2 = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.convert_to_laplacian_mat(dropped_adj2)).cuda()

            perturbed_mat2,_,_ = model.graph_add_edge(sim_dict)
            
            for n, batch in enumerate(adaptive_next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx,lt_idx = batch
                 
                rec_user_emb, rec_item_emb  = model(perturbed_adj=perturbed_mat2,perturbed=False)
                # rec_user_emb2, rec_item_emb2, cl_user_emb2, cl_item_emb2  = model(perturbed=True)
                cl_loss_sgl = model.cal_cl_loss_sgl([user_idx,pos_idx],dropped_adj1,dropped_adj2)
                cl_loss_sgl = self.cl_rate *cl_loss_sgl

                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = adaptive_bpr_loss_downsampling(user_emb, pos_item_emb, neg_item_emb,lt_idx)
                # cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx],rec_user_emb,cl_user_emb,rec_item_emb,cl_item_emb) * 0.5

                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb)  + cl_loss_sgl 
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss_sgl.item(),)
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

        model._load_model(self.user_emb,self.item_emb)
        torch.save(model.state_dict(), f"./model_cpt/galr_{self.config['dataset']}_reweight.pt")



    def get_potential_list(self):
        model = self.model.cuda()
        model.load_state_dict(torch.load(f"./model_cpt/galr_{self.config['dataset']}.pt"))
        self.user_emb, self.item_emb = model._get_embedding()
        model._load_model(self.user_emb,self.item_emb)
        item_acc = self.get_item_performance()

        model = self.model.cuda()
        model.load_state_dict(torch.load(f"./model_cpt/galr_{self.config['dataset']}_reweight.pt"))
        self.user_emb, self.item_emb = model._get_embedding()
        model._load_model(self.user_emb,self.item_emb)
        item_acc2 = self.get_item_performance()

        # 对比 找出有improve.的
        potential_item = {}
        potential_item_inv = {}

        impro_list = []
        for i in item_acc2:
            if item_acc2[i]-item_acc[i] > 0.000001:
                # potential_item.append(i)
                impro_list.append(item_acc2[i]-item_acc[i])

        max_impro = max(impro_list)
        for i in item_acc2:
            if item_acc2[i]-item_acc[i] > 0.000001:
                potential_item[i] = max(0.5, (item_acc2[i]-item_acc[i])/max_impro)
                potential_item_inv[i] = max(0.5,1- (item_acc2[i]-item_acc[i])/max_impro)


        pickle.dump(potential_item,open("{}/potential_item.pkl".format(self.dataset_dir), "wb")) 
        pickle.dump(potential_item_inv,open("{}/potential_item_inv.pkl".format(self.dataset_dir), "wb")) 
        print('num of augmetation', len(impro_list))

    

    def train(self):
        import os
        if not os.path.exists("{}/potential_item_inv.pkl".format(self.dataset_dir)):
            if not os.path.exists(f"./model_cpt/galr_{self.config['dataset']}.pt"):
                self.train_normal()
            self.train_reweighting()
            self.get_potential_list()


        self.model = GALR_Encoder(self.data, self.emb_size, self.eps, self.n_layers,self.layer_cl)

        model = self.model.cuda()

        if 'load_model' in self.train_mode:
            model.load_state_dict(torch.load(f"./model_cpt/galr_{self.config['dataset']}.pt"))
            self.user_emb, self.item_emb = model._get_embedding()
            model._load_model(self.user_emb,self.item_emb)
            print('load model')


        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        

        potential_item=pickle.load(open("{}/potential_item_inv.pkl".format(self.dataset_dir), "rb")) 

        
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
        if not os.path.exists("{}/sim_dict_galr.pkl".format(self.dataset_dir)):
            a = self.data.interaction_mat.T * self.data.interaction_mat 
            a = a.toarray()
            sim_dict = {}
            for i in range(a.shape[0]):
                if not i in popular_item_list:
                    sorted_arr = sorted(range(len(a[i])), key=lambda k: a[i][k])[::-1]
                    sim_dict[i] = np.array(sorted_arr)[np.in1d(np.array(sorted_arr), np.array(popular_item_list))]
            pickle.dump(sim_dict,open("{}/sim_dict_galr.pkl".format(self.dataset_dir), "wb")) 
        else:
            sim_dict = pickle.load(open("{}/sim_dict_galr.pkl".format(self.dataset_dir), "rb"))
        

        self.maxEpoch=65
        for epoch in range(self.maxEpoch):
            # dropped_adj1 = model.graph_reconstruction()
            # dropped_adj2 = model.graph_reconstruction()

            dropped_adj1=GraphAugmentor.edge_dropout_GALR(self.data.interaction_mat, self.drop_rate, user_lc_dict_, item_lc_dict_)
            dropped_adj1 = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.convert_to_laplacian_mat(dropped_adj1)).cuda()
            

            dropped_adj2=GraphAugmentor.edge_dropout_GALR(self.data.interaction_mat, self.drop_rate, user_lc_dict_, item_lc_dict_)
            dropped_adj2 = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.convert_to_laplacian_mat(dropped_adj2)).cuda()


            perturbed_mat2, item_a,item_b = model.graph_add_edge(sim_dict,edge_num = 5)

            self.msg = f"GALR{self.config['dataset']} "
            

            if 'data_mixup' in self.train_mode:        
                for n, batch in enumerate(selective_next_batch_pairwise(self.data, self.batch_size,potential_item,item_a,item_b )):
                    user_idx, pos_idx, neg_idx,lt_idx,item_a2,item_b2 = batch

                    # print(lt_idx,user_pos,item_a,item_b,item_neg )
                    
                    rec_user_emb, rec_item_emb  = model(perturbed_adj=perturbed_mat2,perturbed=False)
                    
                    user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]

                    # rec_user_emb2, rec_item_emb2, cl_user_emb2, cl_item_emb2  = model(perturbed=True)
                    cl_loss_sgl = model.cal_cl_loss_sgl([user_idx,pos_idx],dropped_adj1,dropped_adj2)
                    cl_loss_sgl = self.cl_rate *cl_loss_sgl

                    pos_item_emb1,pos_item_emb2,  = rec_item_emb[item_a2], rec_item_emb[item_b2]
                    pos_item_emb_ = 0.9*pos_item_emb1 + pos_item_emb2*0.1

                    # print(user_emb_.shape, rec_user_emb.shape, len(lt_idx) )

                    # user_emb = torch.cat([user_emb_,user_emb],0)
                    pos_item_emb = torch.cat([pos_item_emb_,pos_item_emb],0)
                    # neg_item_emb = torch.cat([neg_item_emb_,neg_item_emb],0)
                    # sys.exit(1)
                    # print(user_emb.shape, len(lt_idx), neg_item_emb.shape)
                    # print(lt_idx)
                    

                    rec_loss = bpr_loss_galr(user_emb, pos_item_emb, neg_item_emb,lt_idx)
                    # cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx],rec_user_emb,cl_user_emb,rec_item_emb,cl_item_emb) * 0.5

                    batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb)  + cl_loss_sgl 
                    # Backward and optimize
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    if n % 100==0 and n>0:
                        print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss_sgl.item(),)
                with torch.no_grad():
                    self.user_emb, self.item_emb = self.model()
                self.fast_evaluation(epoch)
            else:
                print('sdnjansdianijnsiad')
                for n, batch in enumerate(galr_next_batch_pairwise(self.data, self.batch_size,potential_item)):
                    user_idx, pos_idx, neg_idx,lt_idx = batch

                    # print(lt_idx,user_pos,item_a,item_b,item_neg )
                    
                    rec_user_emb, rec_item_emb  = model(perturbed_adj=perturbed_mat2,perturbed=False)
                    
                    user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]

                    # rec_user_emb2, rec_item_emb2, cl_user_emb2, cl_item_emb2  = model(perturbed=True)
                    cl_loss_sgl = model.cal_cl_loss_sgl([user_idx,pos_idx],dropped_adj1,dropped_adj2)
                    cl_loss_sgl = self.cl_rate *cl_loss_sgl

                    # pos_item_emb1,pos_item_emb2,  = rec_item_emb[item_a2], rec_item_emb[item_b2]
                    # pos_item_emb_ = 0.9*pos_item_emb1 + pos_item_emb2*0.1

                    # print(user_emb_.shape, rec_user_emb.shape, len(lt_idx) )

                    # user_emb = torch.cat([user_emb_,user_emb],0)
                    # pos_item_emb = torch.cat([pos_item_emb_,pos_item_emb],0)
                    # neg_item_emb = torch.cat([neg_item_emb_,neg_item_emb],0)
                    # sys.exit(1)
                    # print(user_emb.shape, len(lt_idx), neg_item_emb.shape)
                    # print(lt_idx)
                    

                    rec_loss = bpr_loss_galr(user_emb, pos_item_emb, neg_item_emb,lt_idx)
                    # rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)

                    # cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx],rec_user_emb,cl_user_emb,rec_item_emb,cl_item_emb) * 0.5

                    batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss_sgl 
                    # Backward and optimize
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    if n % 100==0 and n>0:
                        print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss_sgl.item(),)
                with torch.no_grad():
                    self.user_emb, self.item_emb = self.model()
                self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

        if 'save_model' in self.train_mode:
            model._load_model(self.user_emb,self.item_emb)
            torch.save(model.state_dict(), f'./model_cpt/GALR_{self.config["dataset"]}.pt')


    def cal_cl_loss(self, idx, user_view1,user_view2,item_view1,item_view2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_cl_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss


    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()


class GALR_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, layer_cl):
        super(GALR_Encoder, self).__init__()
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

        # initializer = nn.init.xavier_uniform_
        # virtual_node_emb = (initializer(torch.empty(1, self.latent_size)))
        # print(type(virtual_node_emb), type(item_emb))

        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(user_emb),
            'item_emb': nn.Parameter(item_emb)
            # 'new_item_emb': nn.Parameter( torch.cat( (torch.Tensor(item_emb),virtual_node_emb),0 ) )# append
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
            # 第二个是15和5 第三个10 10

        

        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()

    # def dropped_mat_to_tensor()

    def graph_add_edge(self,sim_dict, edge_num = 5, n_cluster = 3):
        sp_adj = self.data.interaction_mat
        adj_shape = sp_adj.get_shape()
        edge_count = sp_adj.count_nonzero()
        row_idx, col_idx = sp_adj.nonzero()
        # keep_idx = random.sample(range(edge_count), int(edge_count * (1 - drop_rate)))
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        
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
            rdm_itm = item_sim_dict[i][:edge_num]
            for rdm_itm_ in rdm_itm:
                # if rdm_itm_ in self.data.item:
                if clu_result[ i ]==clu_result[ rdm_itm_ ]:
                    # print(user_np.shape)
                    item_a.append( i )
                    item_b.append(  rdm_itm_ )

        user_np = np.append(user_np, item_a)
        item_np = np.append(item_np, item_b)
        item_np = item_np+self.data.user_num
        edges = np.ones_like(user_np, dtype=np.float32)

        n_nodes = self.data.user_num + self.data.item_num

        try:
            dropped_adj = sp.csr_matrix((edges, (user_np, item_np)), shape=(n_nodes, n_nodes), dtype=np.float32)

            # tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep)),shape=,)
        except:
            print(item_a, item_b)
            print(max(user_np), max(item_np))
            sys.exit(1)

        dropped_mat = self.data.convert_to_laplacian_mat_galr(dropped_adj)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda(), item_a,item_b


    def cal_cl_loss_sgl(self, idx, perturbed_mat1, perturbed_mat2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()

        user_view_1, item_view_1,_,__ = self.forward(perturbed=True,perturbed_adj=perturbed_mat1)
        user_view_2, item_view_2,_,__ = self.forward(perturbed=True,perturbed_adj=perturbed_mat2)
        view1 = torch.cat((user_view_1[u_idx],item_view_1[i_idx]),0)
        view2 = torch.cat((user_view_2[u_idx],item_view_2[i_idx]),0)
        # user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        # item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        #return user_cl_loss + item_cl_loss

        # u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        # i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        # user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        # item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        # return 
        return InfoNCE(view1,view2,self.temp)


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