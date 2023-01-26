from base.recommender import Recommender
from data.ui_graph import Interaction
from util.algorithm import find_k_largest
from time import strftime, localtime, time
from data.loader import FileIO
from os.path import abspath
from util.evaluation import ranking_evaluation,get_item_per
import sys
from matplotlib import pyplot as plt
import pickle
import math
from statistics import mean


class GraphRecommender(Recommender):
    def __init__(self, conf, training_set, test_set,test_set1,test_set2,valid_set, **kwargs):
        super(GraphRecommender, self).__init__(conf, training_set, test_set, **kwargs)
        self.data = Interaction(conf, training_set, test_set, mode='train')
        self.bestPerformance = []
        self.bestPerformance2 = []
        top = self.ranking['-topN'].split(',')
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)
        self.valid_data = Interaction(conf, training_set, valid_set)
        self.data1 = Interaction(conf, training_set, test_set1)
        self.data2 = Interaction(conf, training_set, test_set2)
        self.msg = ''


    def print_model_info(self):
        super(GraphRecommender, self).print_model_info()
        print('Training Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.training_size()))
        print('Test Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.test_size()))
        print('=' * 80)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def predict2(self, u):
        pass

    def test_for_t(self):
        inner_data = self.data

        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        user_count = len(inner_data.test_set)
        for i, user in enumerate(inner_data.test_set):
            candidates = self.predict(user)
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            rated_list, li = inner_data.user_rated(user)
            for item in rated_list:
                candidates[inner_data.item[item]] = -10e8
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [inner_data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)

        process_bar(user_count, user_count)
        print('')
        return rec_list


    def test(self):
        inner_data = self.valid_data
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        user_count = len(inner_data.test_set)
        for i, user in enumerate(inner_data.test_set):
            # print(i, user)
            candidates = self.predict(user)

            rated_list, li = inner_data.user_rated(user)

            for item in rated_list:
                candidates[inner_data.item[item]] = -10e8

            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [inner_data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))

            if i % 1000 == 0:
                process_bar(i, user_count)


        process_bar(user_count, user_count)
        print('')
        return rec_list

    
    def test2(self):
        inner_data = self.valid_data

        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        user_count = len(inner_data.test_set)
        for i, user in enumerate(inner_data.test_set):
            candidates = self.predict2(user)
            rated_list, li = inner_data.user_rated(user)
            for item in rated_list:
                candidates[inner_data.item[item]] = -10e8
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [inner_data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)

        process_bar(user_count, user_count)
        print('')
        return rec_list


    def test_for_eval_dual(self,inner_data):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        user_count = len(inner_data.test_set)
        for i, user in enumerate(inner_data.test_set):
            candidates = self.predict(user)
            candidates2 = self.predict2(user)
            candidates = candidates + candidates2
            candidates = candidates/2

            rated_list, li = inner_data.user_rated(user)
            for item in rated_list:
                candidates[inner_data.item[item]] = -10e8
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [inner_data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list


    def test_for_eval(self,inner_data):

        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        user_count = len(inner_data.test_set)
        for i, user in enumerate(inner_data.test_set):
            candidates = self.predict(user)
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            rated_list, li = inner_data.user_rated(user)
            for item in rated_list:
                candidates[inner_data.item[item]] = -10e8
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [inner_data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)

            
        process_bar(user_count, user_count)
        print('')
        return rec_list

    def test_for_eval_neg_sam(self,inner_data):
 

        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        user_count = len(inner_data.test_set)
        for i, user in enumerate(inner_data.test_set):
            candidates = self.predict(user)
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            rated_list, li = inner_data.user_rated(user)
            for item in rated_list:
                candidates[inner_data.item[item]] = -10e8

            # negative sampling 100
            item_str = set(list(inner_data.id2item.values()))
            testset_item_str = set(list(inner_data.test_set[user].keys()))
            trainset_item_str = set(rated_list)
            import random
            random_sample_100_item = random.sample((item_str - testset_item_str - trainset_item_str),100)
            for item in random_sample_100_item:
                candidates[inner_data.item[item]] += 1000
            for item in testset_item_str:
                candidates[inner_data.item[item]] += 1000

            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [inner_data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)

        process_bar(user_count, user_count)
        print('')
        return rec_list

    

    def evaluate(self):

        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))

        out_dir = self.output['-dir']

        file_name = self.config['model.name'] + '@' + current_time + '-performance' + '.txt'
        rec_list = self.test_for_eval(self.data)
        self.result = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        self.model_log = ''
        self.model_log +=('###Evaluation Results###\n')
        self.model_log+= str(self.bestPerformance[0])
        self.model_log+=' '.join(self.result)
        
        print('The result of %s:\n%s' % (self.model_name, ''.join(self.result)))

        self.model_log+=('\n###Evaluation Results for head###\n')
        rec_list = self.test_for_eval(self.data1)
        self.result = ranking_evaluation(self.data1.test_set, rec_list, self.topN)
        self.model_log+=' '.join(self.result)

        self.model_log+=('\n###Evaluation Results for tail###\n')
        rec_list = self.test_for_eval(self.data2)
        self.result = ranking_evaluation(self.data2.test_set, rec_list, self.topN)
        self.model_log+=' '.join(self.result)
        self.model_log+=(f"\n###  {self.config['model.name']} {self.msg} ###")
        FileIO.write_file(out_dir, file_name, self.model_log)



    def get_item_performance(self):
        print('get_item_performance...')
        rec_list = self.test()
        item_acc,_ = get_item_per(self.valid_data.test_set, rec_list)
        return item_acc

    def select_poor_performance_item(self):
        print('select_poor_performance_item...')
        rec_list = self.test_for_t()
        _,item_dict = get_item_per(self.data.test_set, rec_list)

        for i in item_dict:
            item_dict[i] = mean(item_dict[i])

        data_val_count = self.data.train_value_count
        result_dict = {}
        select_type = 'prop'
        threshold =0.4
        if select_type == 'prop':
            from collections import OrderedDict
            item_dict_od = OrderedDict(sorted(item_dict.items(), key=lambda x: x[1]))
            poor_items = list(item_dict_od.keys())[:int(0.8* len(item_dict))]
        else:
            poor_items = list()
            for i in item_dict:
                if item_dict[i] < threshold:
                    poor_items.append(i)

        return poor_items


    def draw_item_performance_figure(self,file_name='',print_fig=True):
        print('draw_item_performance_figure...')
        rec_list = self.test_for_t()
        _,item_dict = get_item_per(self.data.test_set, rec_list)

        data_val_count = self.data.train_value_count

        result_dict = {}
        result_dict_smooth = {}
        result_log_dict = {}

        print(data_val_count)

        for i in data_val_count.iteritems():
            val = i[1]
            item_name = i[0]
            temp = result_dict.get(val, list())
            temp2 = item_dict.get(item_name, list())
            if val > 1000:
                var = val - val%1000
            elif val > 100:
                var = val - val%100
            elif val > 10:
                var = val - val%10
            else:
                var = val

            result_dict[var] =temp + temp2

        for key, value in result_dict.items():
            result_log_dict[(key+1)] = mean(value)

        plt.plot(list(result_log_dict.keys()), list(result_log_dict.values()), label=file_name)
        plt.xscale('log')
        plt.legend(loc='upper left')
        
        if print_fig:
            plt.savefig('./fig/'+file_name+'.png')
            pickle.dump(result_log_dict, open("./fig/{}_result_dic.pkl".format(file_name), "wb"))

        return result_log_dict

    
    def fast_evaluation(self, epoch):
        print('evaluating the model...')
        rec_list = self.test()

        measure = ranking_evaluation(self.valid_data.test_set, rec_list, [self.max_N])
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance[1] = performance
                self.bestPerformance[0] = epoch + 1
                self.save()
        else:
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
                self.bestPerformance.append(performance)
            self.save()
        print('-' * 120)
        print('Real-Time Ranking Performance ' + ' (Top-' + str(self.max_N) + ' Item Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', ' | '.join(measure))
        bp = ''

        bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + ' | '

        bp += 'NDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
        print('*Best Performance* ')
        print('Epoch:', str(self.bestPerformance[0]) + ',', bp)
        print('-' * 120)

        return measure


    def fast_evaluation2(self, epoch, model):
        print('evaluating the model...')
        rec_list = self.test2()

        measure = ranking_evaluation(self.valid_data.test_set, rec_list, [self.max_N])
        if len(self.bestPerformance2) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            for k in self.bestPerformance2[1]:
                if self.bestPerformance2[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance2[1] = performance
                self.bestPerformance2[0] = epoch + 1
                self.save2(model)
        else:
            self.bestPerformance2.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
                self.bestPerformance2.append(performance)
            self.save2(model)
        print('-' * 120)
        print('Real-Time Ranking Performance ' + ' (Top-' + str(self.max_N) + ' Item Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', ' | '.join(measure))
        bp = ''

        bp += 'Recall' + ':' + str(self.bestPerformance2[1]['Recall']) + ' | '

        bp += 'MDCG' + ':' + str(self.bestPerformance2[1]['NDCG'])
        print('*Best Performance* ')
        print('Epoch:', str(self.bestPerformance2[0]) + ',', bp)
        print('-' * 120)

        return measure
