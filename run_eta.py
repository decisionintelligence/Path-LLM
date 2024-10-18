import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
import torch
from exp import exp_ETA
import time
import logging

fix_seed = 2024
torch.manual_seed(fix_seed)

parser = argparse.ArgumentParser(description='LLM4Path')
parser.add_argument('--data_path',type=str)
parser.add_argument('--model',type=str,default="GPT4Path")
parser.add_argument('--road_size', type=int, default=4315, help='the number of path')
parser.add_argument('--embedding_dim', type=int, default=768,help='the dimension of embedding')
parser.add_argument('--out_dim',type=int,default=768,help='the dimension of model output')
parser.add_argument('--max_len', type=int, default=515)
parser.add_argument('--batch_size',type=int,default=24)
parser.add_argument('--learning_rate',type=float,default=0.001)
parser.add_argument('--train_epochs',type=int,default=1)
parser.add_argument('--lradj',type=str,default='type1')
parser.add_argument('--data_name',type=str,default='chengdu',help='the name of data')
parser.add_argument('--is_load_pretrain_model',type=int,default=0,help='the choice of loading pretrain model params')
parser.add_argument('--node2vec_embedding',type=str,default=None,help='the path of node2vec embedding')

args = parser.parse_args()

print('Args in experiment:')
print(args)
Exp_dict = {
            'PathLLM':exp_ETA.Exp_PathLLM,
        }
Exp = Exp_dict[args.model]


exp = Exp(args)  # set experiments



#几月几日+几点几分+数据名+batch_size:0318_2034_chengdu_24
mon_day = "1016"
min_sec = '0200'
setting = '{}_{}_{}_bts{}_shiyan'.format(
        mon_day,
        min_sec,
        args.data_name,
        args.batch_size,
        )

logging.basicConfig(filename='./logs/'+setting+'_execution.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

print(setting)
print('>>>>>>> start training >>>>>>>>>>>>>>')
exp.train(setting,logging)

print('>>>>>>> testing <<<<<<<<<<<<<<<<<<')
test_time = time.time()
exp.test(setting)
logging.info(f'the cost of test data:{test_time-time.time():.3f}秒.')
torch.cuda.empty_cache()
 