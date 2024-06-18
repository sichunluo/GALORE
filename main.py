from Rec import Rec
from util.conf import ModelConf
import argparse
import time

if __name__ == '__main__':
    model = 'GALORE'
    s = time.time()
    conf = ModelConf('./conf/' + model + '.conf')
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_mode", type=str, default='train_normal', help='')
    parser.add_argument("--dataset", type=str, default='ml-1M', help='')
    parser.add_argument("--emb", type=str, default='64', help='')
    parser.add_argument("--add_edge", type=str, default='3', help='')
    parser.add_argument("--drop_rate", type=str, default='0.1', help='')
    parser.add_argument("--cluster_num", type=str, default='1', help='')


    args = parser.parse_args()
    conf.__setitem__('train_mode',args.train_mode)
    conf.__setitem__('dataset',args.dataset)
    conf.__setitem__('training.set',f'./dataset/{args.dataset}/train.txt')
    conf.__setitem__('valid.set',f'./dataset/{args.dataset}/valid.txt')
    conf.__setitem__('test.set',f'./dataset/{args.dataset}/test.txt')
    conf.__setitem__('test1.set',f'./dataset/{args.dataset}/test1.txt')
    conf.__setitem__('test2.set',f'./dataset/{args.dataset}/test2.txt')

    conf.__setitem__('embbedding.size',args.emb)
    conf.__setitem__('drop_rate',args.drop_rate)
    conf.__setitem__('add_edge',args.add_edge)
    conf.__setitem__('cluster_num',args.cluster_num)



    rec = Rec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
