from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate,get_batch_mask
import torch
import torch.nn as nn
from torch import optim
from sklearn import preprocessing
from transformers import GPT2Tokenizer
import os
import time
import warnings
import numpy as np
warnings.filterwarnings('ignore')
class Exp_Gpt_prompt(Exp_Basic):

    def __init__(self, args):
        super(Exp_Gpt_prompt, self).__init__(args)
        self.max_len = args.max_len
        
        # self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # if self.tokenizer.eos_token:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        # else:
        #     pad_token = '[PAD]'
        #     self.tokenizer.add_special_tokens({'pad_token': pad_token})
        #     self.tokenizer.pad_token = pad_token
    def _build_model(self):
        # text_embeddings = np.load('./xian_dataset/text_embeddings_xian.npy')
        
        # last_row = text_embeddings[-1,:].reshape(1,-1)  
        # expanded_array = np.repeat(last_row,repeats=4316-len(text_embeddings), axis=0)
        # text_embeddings = np.vstack([text_embeddings, expanded_array])       
        # text_embeddings = torch.tensor(text_embeddings).cuda()
        
        # prompt_embedding = np.load('./dataset/prompt_embedding.npy')
        # prompt_embedding = torch.tensor(prompt_embedding).cuda()
        
        # road_embeddings = np.load('./xian_dataset/node2vec_embedding_xian.npy')
        

        # last_row = road_embeddings[-1,:].reshape(1,-1)  
        # expanded_array = np.repeat(last_row,repeats=4316-len(road_embeddings), axis=0)
        # road_embeddings = np.vstack([road_embeddings, expanded_array])       
        # road_embeddings = torch.tensor(road_embeddings).cuda()

        text_embeddings = np.load('./xian_dataset/text_embeddings_xian.npy')
        text_embeddings = torch.tensor(text_embeddings).cuda()
        prompt_embedding = np.load('./dataset/prompt_embedding.npy')
        prompt_embedding = torch.tensor(prompt_embedding).cuda()
        
        road_embeddings = np.load('./xian_dataset/node2vec_embedding_xian.npy')
        road_embeddings = torch.tensor(road_embeddings).cuda()
        

        model_para = torch.load('./experiments/Gpt_prompt/checkpoints/0823_3200_chengdu_bts8_shiyan/checkpoint.pth')
        for key in list(model_para.keys()):
            
            if 'road_embeddings' in key or 'text_embed' in key:
                print(key)
                model_para[key[7:]] = model_para[key] #全部key去掉“module.”前缀
                del model_para[key]

        
        model = self.model_dict[self.args.model](self.args,text_embeddings,prompt_embedding,road_embeddings).float()
        model.load_state_dict(model_para,strict = False)
        

        # model.load_state_dict(model_data)
        # if self.args.is_load_pretrain_model == 1:
        #     param = torch.load('./params/GPTmodel.pt')
        #     model.load_state_dict(param)
        
        # if self.args.use_multi_gpu and self.args.use_gpu:
        #     model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        print(flag)
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        
        total_loss = []
        total_mape = []
        total_mae = []
        self.model.eval()
        with torch.no_grad():
            for data in vali_loader:
                # val_roads,val_time,val_valid_len,val_prompt = data
                val_roads,val_time,val_valid_len = data

                #val_prompt = self.tokenizer(val_prompt, return_tensors="pt", padding=True,truncation=True, max_length=500).input_ids.cuda()  
                mask = get_batch_mask(len(val_valid_len),val_roads.size(1),valid_len=val_valid_len)
                mask.to(torch.float).cuda()
                pred = self.model(val_roads, mask, val_valid_len)
                loss = criterion(pred, val_time)
                mape = torch.mean(torch.abs((pred - val_time) / val_time)).item()
                mae = torch.mean(torch.abs(pred-val_time)).item()

                total_loss.append(loss.item())   
                total_mape.append(mape)  
                total_mae.append(mae)

        total_loss = np.average(total_loss)
        total_mape = np.average(total_mape)
        total_mae = np.average(total_mae)

        self.model.train()
        return total_loss,total_mape,total_mae

    
    def test(self,setting):

        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        # path = os.path.join('./experiments/Gpt_prompt/checkpoints',setting)
        # model_path = './experiments/Gpt_prompt/checkpoints/0415_2010_chengdu_bts24_concat(node+text)'
        # print('loading model')
        # pretrained_dict = torch.load(os.path.join(model_path, 'checkpoint.pth'))
        # print(type(pretrained_dict))
        
        # model_dict = self.model.state_dict()
        # pretrained_dict = {key: value for key, value in pretrained_dict.items() if (key in pretrained_dict and ('text_embeddings' not in key and 'road_embeddings' not in key ))}
        
        # model_dict.update(pretrained_dict)
        # print(type(model_dict))
        # self.model.load_state_dict(model_dict,strict=False)
        
        best_model_train_loss = []
        best_model_test_loss = []
        best_model_train_mape = []
        best_model_test_mape = []
        best_model_train_mae = []
        best_model_test_mae = []
        best_model_train_mare = []
        best_model_test_mare = []
        folder_path = './transfer_experiments/Gpt_prompt/results/' + setting + '/' + 'test/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # self.model.eval()
        criterion = self._select_criterion()
        print('----------starting testing train data----------')

        # (1) stastic on the train set
        with torch.no_grad():
            for data in train_loader:
                train_roads,train_time,train_valid_len = data
                # train_prompt = self.tokenizer(train_prompt, return_tensors="pt", padding=True,truncation=True, max_length=500).input_ids.cuda()
                mask = get_batch_mask(len(train_valid_len),train_roads.size(1),train_valid_len)
                mask.to(torch.float).cuda()
                pred = self.model(train_roads,mask,train_valid_len)
                # criterion
                loss = criterion(pred, train_time)
                mape = torch.mean(torch.abs((pred - train_time) / train_time)).item()
                mae = torch.mean(torch.abs(pred-train_time)).item()
                mare = (torch.sum(torch.abs(pred - train_time))/torch.sum(train_time)).item()
                best_model_train_loss.append(loss.item())
                best_model_train_mape.append(mape)
                best_model_train_mae.append(mae)
                best_model_train_mare.append(mare)

        #loading results
        best_model_train_results = []
        best_model_train_results.append(best_model_train_loss)
        best_model_train_results.append(best_model_train_mape)
        best_model_train_results.append(best_model_train_mae)
        best_model_train_results.append(best_model_train_mare)
        np.save(folder_path+'best_model_train_results.npy',best_model_train_results)

       

        print('----------starting testing test data----------')
        with torch.no_grad():
            for data in test_loader:
                test_roads,test_time,test_valid_len = data
                # test_prompt = self.tokenizer(test_prompt, return_tensors="pt", padding=True,truncation=True, max_length=500).input_ids.cuda()
                mask = get_batch_mask(len(test_valid_len),test_roads.size(1),test_valid_len)
                mask.to(torch.float).cuda()
                pred = self.model(test_roads,mask,test_valid_len)

                loss = criterion(pred,test_time)
                mape = torch.mean(torch.abs((pred - test_time) / test_time)).item()
                mae = torch.mean(torch.abs(pred-test_time)).item()
                mare = (torch.sum(torch.abs(pred - test_time))/torch.sum(test_time)).item()
                best_model_test_loss.append(loss.item())
                best_model_test_mape.append(mape)
                best_model_test_mae.append(mae)
                best_model_test_mare.append(mare)

        #loading results
        best_model_test_results = []
        best_model_test_results.append(best_model_test_loss)
        best_model_test_results.append(best_model_test_mape)
        best_model_test_results.append(best_model_test_mae)
        best_model_test_results.append(best_model_test_mare)
        np.save(folder_path+'best_model_test_results.npy',best_model_test_results)

        return