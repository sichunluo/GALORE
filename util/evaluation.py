import math
from statistics import mean 

class Metric(object):
    def __init__(self):
        pass

    
    @staticmethod
    def hits_item(origin, res):
        item_dict = {}

        item_dict_occ = {}
        item_dict_acc ={}

        for user in origin:
            predicted = [item[0] for item in res[user]]
            for item in list(origin[user].keys()):
                temp = item_dict.get(item,list())
                temp.append( len(set([item]).intersection(set(predicted))) )
                item_dict[item] = temp

        for i in item_dict:
            item_dict_occ[i] = len(item_dict[i])
            item_dict_acc[i] = mean(item_dict[i])

        return item_dict_acc,item_dict


    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hit_count[user] = len(set(items).intersection(set(predicted)))
            # set(head_items)
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        """
        Note: This type of hit ratio calculates the fraction:
         (# retrieved interactions in the test set / #all the interactions in the test set)
        """
        total_num = 0
        for user in origin:
            items = list(origin[user].keys())
            total_num += len(items)
        hit_num = 0
        for user in hits:
            hit_num += hits[user]
        return round(hit_num/total_num,5)


    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return round(prec / (len(hits) * N),5)


    @staticmethod
    def recall(hits, origin):
        recall_list = [hits[user]/len(origin[user]) for user in hits]
        recall = sum(recall_list) / len(recall_list)
        return round(recall,5)

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return round(2 * prec * recall / (prec + recall),5)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error+=abs(entry[2]-entry[3])
            count+=1
        if count==0:
            return error
        return error/count

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[2] - entry[3])**2
            count += 1
        if count==0:
            return error
        return math.sqrt(error/count)

    @staticmethod
    def NDCG(origin,res,N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            #1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG+= 1.0/math.log(n+2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG+=1.0/math.log(n+2)
            sum_NDCG += DCG / IDCG
        return round(sum_NDCG / len(res),5)

def get_item_per(origin, res):
    item_dict_acc,item_dict = Metric.hits_item(origin, res)
    return item_dict_acc,item_dict

def ranking_evaluation(origin, res, N):
    measure = []
    for n in N:
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        indicators = []
        if len(origin) != len(predicted):
            print(len(origin) , len(predicted))
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)

        hits = Metric.hits(origin, predicted)

        recall = Metric.recall(hits, origin)
        
        indicators.append('Recall:' + str(recall) + '\n')

        NDCG = Metric.NDCG(origin, predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')

        measure.append('Top ' + str(n) + '\n')
        measure += indicators
    return measure

def rating_evaluation(res):
    measure = []
    mae = Metric.MAE(res)
    measure.append('MAE:' + str(mae) + '\n')
    rmse = Metric.RMSE(res)
    measure.append('RMSE:' + str(rmse) + '\n')
    return measure