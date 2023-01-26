from Rec import Rec
from util.conf import ModelConf
import argparse
import time

if __name__ == '__main__':
    model = 'GALR'
    s = time.time()
    conf = ModelConf('./conf/' + model + '.conf')
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_mode", type=str, default='train_normal', help='')
    parser.add_argument("--dataset", type=str, default='ml-1M', help='')
    args = parser.parse_args()
    conf.__setitem__('train_mode',args.train_mode)
    conf.__setitem__('dataset',args.dataset)
    conf.__setitem__('training.set',f'./dataset/{args.dataset}/train.txt')
    conf.__setitem__('valid.set',f'./dataset/{args.dataset}/valid.txt')
    conf.__setitem__('test.set',f'./dataset/{args.dataset}/test.txt')
    conf.__setitem__('test1.set',f'./dataset/{args.dataset}/test1.txt')
    conf.__setitem__('test2.set',f'./dataset/{args.dataset}/test2.txt')
    rec = Rec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
