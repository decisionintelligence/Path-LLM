from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate,get_batch_mask
import torch
import torch.nn as nn
from torch import optim
from transformers import GPT2Tokenizer
import os
import time
import warnings
import numpy as np
warnings.filterwarnings('ignore')
class Exp_PathLLM(Exp_Basic):
    def __init__(self, args):
        super(Exp_PathLLM, self).__init__(args)
        self.max_len = args.max_len
        
        # self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # if self.tokenizer.eos_token:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        # else:
        #     pad_token = '[PAD]'
        #     self.tokenizer.add_special_tokens({'pad_token': pad_token})
        #     self.tokenizer.pad_token = pad_token
    def _build_model(self):
        # xi an
        text_embeddings = np.load('./xian_dataset/text_embeddings_xian_pattern.npy')
        text_embeddings = torch.tensor(text_embeddings).cuda()
        prompt_embedding = np.load('./dataset/prompt_embedding.npy')
        prompt_embedding = torch.tensor(prompt_embedding).cuda()
        road_embeddings = np.load('./xian_dataset/node2vec_embedding_xian.npy')
        road_embeddings = torch.tensor(road_embeddings).cuda()
        
        # print(road_embeddings.size())
        
        # print(text_embeddings.size())

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
        
        model = self.model_dict[self.args.model](self.args,text_embeddings,prompt_embedding,road_embeddings).float()
        if self.args.is_load_pretrain_model == 1:
            param = torch.load('./params/GPTmodel.pt')
            model.load_state_dict(param)
        
        # if self.args.use_multi_gpu and self.args.use_gpu:
        #     model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag,st=0,end=0):
        print(flag)
        data_set, data_loader = data_provider(self.args, flag,st,end)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,weight_decay=1e-3)
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
                val_roads,val_time,val_valid_len = data
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

    def train(self,setting,logging):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        train_batch_num = len(train_loader)
        
        
        path = os.path.join('./experiments/Gpt_prompt/checkpoints',setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=30, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        train_loss = []
        train_mape = []
        train_mae = []
        test_loss = []
        test_mape = []
        test_mae = []
        val_loss = []
        val_mape = []
        val_mae = []
        iter_count = 0
        self.args.train_epochs
        data_len = len(train_loader)

        for epoch in range(40):
            epoch_start = time.time()
            train_loss_i = 0
            train_mape_i = 0
            train_mae_i = 0
            self.model.train()
            batch_break = 0
            batch_begin = 0
            print(data_len)
            # xian 
            if epoch < 11:
                batch_break = data_len / 3
                batch_begin = 0
            elif epoch >= 11 and epoch < 25:
                batch_break = 2 * data_len / 3 + 1 * data_len / 4
                batch_begin = 1 * data_len / 4
            else:
                batch_break = data_len
                batch_begin = 1 * data_len / 3
            for batch_id,data in enumerate(train_loader):
                
                if(batch_id > batch_break):
                    break
                
                if(batch_id < batch_begin):
                    continue
                batch_st = time.time()
                # train_roads,train_time,train_valid_len,train_prompt = data
                train_roads,train_time,train_valid_len = data
                mask = get_batch_mask(len(train_valid_len),train_roads.size(1),train_valid_len)
                mask.to(torch.float).cuda()
                pred = self.model(train_roads,mask,train_valid_len)
                iter_count += 1
                loss = criterion(pred,train_time)
                mape = torch.mean(torch.abs((pred - train_time) / train_time)).item()
                mae = torch.mean(torch.abs(pred - train_time)).item()
                train_loss_i += loss.item()
                train_mape_i += mape
                train_mae_i += mae

                model_optim.zero_grad()
                loss.backward()
                model_optim.step()
                batch_end = time.time()
                #logging.info(f'the cost of one bacth{batch_end-batch_st:.3f}秒.')
                print("\t train iters: {0}, epoch: {1} | loss: {2:.7f} |   mape:{3:.6f}".format(iter_count , epoch + 1, loss.item(),mape))  

            train_loss_i /= train_batch_num
            train_mape_i /= train_batch_num
            train_mae_i /= train_batch_num

             
            print('----------starting vali----------')
            val_loss_i,val_mape_i,val_mape_i = self.vali(vali_data, vali_loader, criterion)
            val_loss.append(val_loss_i)
            val_mape.append(val_mape_i)
            val_mae.append(val_mape_i)

            print('----------starting test----------')
            test_loss_i,test_mape_i,test_mae_i = self.vali(test_data, test_loader, criterion)
            test_loss.append(test_loss_i)
            test_mape.append(test_mape_i)
            test_mae.append(test_mae_i)

            early_stopping(val_loss_i, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

            epoch_end = time.time()
            logging.info(f'the cost of one epoch:{epoch_end-epoch_start:.3f}秒.')

        save_path = './experiments/Gpt_prompt/results/' + setting + '/' + 'train/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        #mse,mape,mae,mare = res
        train_results = []
        train_results.append(train_loss)
        train_results.append(train_mape)
        train_results.append(train_mae)
        np.save(save_path+'train_results.npy',train_results)
        
        val_results = []
        val_results.append(val_loss)
        val_results.append(val_mape)
        val_results.append(val_mae)
        np.save(save_path+'val_results.npy',val_results)

        test_results = []
        test_results.append(test_loss)
        test_results.append(test_mape)
        test_results.append(test_mae)
        np.save(save_path+'test_results.npy',test_results)



        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        torch.save(self.model,'./params/'+setting+'_Gpt_prompt.pt')
        return self.model

    def test(self,setting):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        path = os.path.join('./experiments/Gpt_prompt/checkpoints',setting)
        
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')),strict = False)
        
        best_model_train_loss = []
        best_model_test_loss = []
        best_model_train_mape = []
        best_model_test_mape = []
        best_model_train_mae = []
        best_model_test_mae = []
        best_model_train_mare = []
        best_model_test_mare = []
        folder_path = './experiments/Gpt_prompt/results/' + setting + '/' + 'test/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        self.model.eval()
        criterion = self._select_criterion()
        # print('----------starting testing train data----------')

        # # (1) stastic on the train set
        # with torch.no_grad():
        #     for data in train_loader:
        #         train_roads,train_time,train_valid_len = data
        #         # train_prompt = self.tokenizer(train_prompt, return_tensors="pt", padding=True,truncation=True, max_length=500).input_ids.cuda()
        #         mask = get_batch_mask(len(train_valid_len),train_roads.size(1),train_valid_len)
        #         mask.to(torch.float).cuda()
        #         pred = self.model(train_roads,mask,train_valid_len)
        #         # criterion
        #         loss = criterion(pred, train_time)
        #         mape = torch.mean(torch.abs((pred - train_time) / train_time)).item()
        #         mae = torch.mean(torch.abs(pred-train_time)).item()
        #         mare = (torch.sum(torch.abs(pred - train_time))/torch.sum(train_time)).item()
        #         best_model_train_loss.append(loss.item())
        #         best_model_train_mape.append(mape)
        #         best_model_train_mae.append(mae)
        #         best_model_train_mare.append(mare)
        # #loading results
        # best_model_train_results = []
        # best_model_train_results.append(best_model_train_loss)
        # best_model_train_results.append(best_model_train_mape)
        # best_model_train_results.append(best_model_train_mae)
        # best_model_train_results.append(best_model_train_mare)
        # np.save(folder_path+'best_model_train_results.npy',best_model_train_results)

       

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