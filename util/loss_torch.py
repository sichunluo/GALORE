import torch
import torch.nn.functional as F
import numpy as np

def distance_loss(feat1,feat2):
    return  -F.cosine_similarity(feat1, feat2).mean()
    

def adaptive_bpr_loss_downsampling(user_emb, pos_item_emb, neg_item_emb,lt_idx):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    lt_idx = -0.5 * np.array(lt_idx) + 1 
    lt_idx_ = torch.Tensor(lt_idx).cuda()
    pos_score = pos_score 
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score))
    loss = loss * lt_idx_
    return torch.mean(loss)

def adaptive_bpr_loss_upsampling(user_emb, pos_item_emb, neg_item_emb,lt_idx):
    print(user_emb.shape, pos_item_emb.shape)
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    lt_idx = -0.25 * np.array(lt_idx) + 1.25
    lt_idx_ = torch.Tensor(lt_idx).cuda()
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score))
    loss = loss * lt_idx_
    return torch.mean(loss) 

def bpr_loss_galr(user_emb, pos_item_emb, neg_item_emb,lt_idx):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    lt_idx = 1.75 * np.array(lt_idx) + 1
    lt_idx_ = torch.Tensor(lt_idx).cuda()
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score))
    loss = loss * lt_idx_
    return torch.mean(loss)  



def bpr_loss_value(user_emb, pos_item_emb, neg_item_emb,lt_idx, a,b):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    n1 = a
    n2= b-a
    lt_idx = -n2 * np.array(lt_idx)  + n1
    lt_idx_ = torch.Tensor(lt_idx).cuda()
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score))
    loss = loss * lt_idx_
    return torch.mean(loss) 


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score))
    return torch.mean(loss)


def triplet_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = F.relu(neg_score+1-pos_score)
    return torch.mean(loss)

def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2)
    return emb_loss * reg


def batch_softmax_loss(user_emb, item_emb, temperature):
    user_emb, item_emb = F.normalize(user_emb, dim=1), F.normalize(item_emb, dim=1)
    pos_score = (user_emb * item_emb).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(user_emb, item_emb.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    loss = -torch.log(pos_score / ttl_score)
    return torch.mean(loss)


def InfoNCE(view1, view2, temperature):
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score)
    return torch.mean(cl_loss)



def kl_divergence(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(kl)

def js_divergence(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    q = F.softmax(q_logit, dim=-1)
    kl_p = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
    kl_q = torch.sum(q * (F.log_softmax(q_logit, dim=-1) - F.log_softmax(p_logit, dim=-1)), 1)
    return torch.mean(kl_p+kl_q)