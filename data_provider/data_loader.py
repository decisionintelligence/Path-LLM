from torch.utils.data import Dataset
import pickle as pkl
import os
import numpy as np
import torch

class Basic_Dataset(Dataset):
    def __init__(self,args,flag,st=0,end=0) :
        trip_time = np.load(os.path.join(args.data_path,'trip_time_remove_10_xian.npy'))/60 
        trip_time = torch.tensor(trip_time).float().cuda()
        #data_road = data_dep[:,:-1]
        data_road = np.load(os.path.join(args.data_path,'data_road_remove_10_xian.npy'))[:,:143]
        data_road = torch.tensor(data_road).cuda()
        #row_num = data_dep[:,-1]
        row_num = np.load(os.path.join(args.data_path,'row_num_remove_10_xian.npy'))
        row_num = torch.tensor(row_num).cuda()
        self.trip_num = len(row_num)
        
        
        #8:1:1
        train_line = int(self.trip_num*0.8)
        val_line = int(self.trip_num*0.9)
        # sort_index_train = np.load(os.path.join(args.data_path,'sort_index_train_xian.npy'))
        self.flag = flag
        self.train_data_road = data_road[0:train_line]
        # self.train_data_road = self.train_data_road[sort_index_train]
        # self.train_data_road = torch.cat([self.train_data_road,self.train_data_road[5000:]],dim=0)

        self.train_trip_time = trip_time[0:train_line]
        # self.train_trip_time = self.train_trip_time[sort_index_train]
        # self.train_trip_time = torch.cat([self.train_trip_time,self.train_trip_time[5000:]],dim=0)

        

        self.train_row_num = row_num[0:train_line]
        # self.train_row_num = self.train_row_num[sort_index_train]
        # self.train_row_num = torch.cat([self.train_row_num,self.train_row_num[5000:]],dim=0)
  
        self.val_data_road = data_road[train_line:val_line]
        self.val_trip_time = trip_time[train_line:val_line]
        self.val_row_num = row_num[train_line:val_line]

        self.test_data_road = data_road[val_line:]
        self.test_trip_time = trip_time[val_line:]
        self.test_row_num = row_num[val_line:]
        
        
        # few-shot
        # self.trip_num = int(len(row_num) * 0.01)
        # train_line = int(self.trip_num*0.8)
        # val_line = int(self.trip_num*0.9)
        # self.train_data_road = data_road[0:train_line]
        # self.train_trip_time = trip_time[0:train_line]
        # self.train_row_num = row_num[0:train_line]
        # self.val_data_road = data_road[train_line:val_line]
        # self.val_trip_time = trip_time[train_line:val_line]
        # self.val_row_num = row_num[train_line:val_line]


    def __getitem__(self, index):
        if self.flag == "train":
            return self.train_data_road[index],self.train_trip_time[index],self.train_row_num[index]
        elif self.flag == "val":
            return self.val_data_road[index],self.val_trip_time[index],self.val_row_num[index]
        else:
            return self.test_data_road[index],self.test_trip_time[index],self.test_row_num[index]
    def __len__(self):
        if self.flag == "train":
            return len(self.train_row_num)
        elif self.flag == "val":
            return len(self.val_row_num)
        else:
            return len(self.test_row_num)
        
class PimDataset(Dataset):
    
    def __init__(self,args,flag):
        
        #Path1,_,_,_,_,_,row_num=pkl.load(open('./dataset/path123_pos1_neg12.pkl','rb'))
        Path1 = np.load('./xian_dataset/data_road_remove_10_xian.npy')
        row_num = np.load('./xian_dataset/row_num_remove_10_xian.npy')
        trip_time = np.load('./xian_dataset/trip_time_remove_10_xian.npy')/60
        trip_time = torch.FloatTensor(trip_time).cuda()
        Path1Mask = np.load('xian_dataset/Path1_Mask_xian.npy')
        self.Path1Mask = torch.FloatTensor(Path1Mask).cuda()
        self.trip_time = trip_time
        self.Path1 = torch.LongTensor(Path1).cuda()
        self.row_num = torch.Tensor(row_num).cuda()
        self.len=Path1.shape[0]
        self.trip_num = len(row_num)
        #8:1:1
        train_line = int(self.trip_num*0.8)
        val_line = int(self.trip_num*0.9)

        self.flag = flag
        #train data
        self.train_Path1 = self.Path1[0:train_line]
        self.train_trip_time = self.trip_time[0:train_line]
        self.train_Path1Mask = self.Path1Mask[0:train_line]
        self.train_row_num = self.row_num[0:train_line]
        #val data
        self.val_Path1 = self.Path1[train_line:val_line]
        self.val_trip_time = self.trip_time[train_line:val_line]
        self.val_Path1Mask = self.Path1Mask[train_line:val_line]
        self.val_row_num = self.row_num[train_line:val_line]
        #test data
        self.test_Path1 = self.Path1[val_line:]
        self.test_trip_time = self.trip_time[val_line:]
        self.test_Path1Mask = self.Path1Mask[val_line:]
        self.test_row_num = self.row_num[val_line:]

    def __getitem__(self,index):
        if self.flag == "train":
            return self.train_Path1[index],self.train_trip_time[index],self.train_Path1Mask[index],self.train_row_num[index]
        elif self.flag == "val":
            return self.val_Path1[index],self.val_trip_time[index],self.val_Path1Mask[index],self.val_row_num[index]
        else:
            return self.test_Path1[index],self.test_trip_time[index],self.test_Path1Mask[index],self.test_row_num[index]
        
    def __len__(self):
        if self.flag == "train":
            return len(self.train_row_num)
        elif self.flag == "val":
            return len(self.val_row_num)
        else:
            return len(self.test_row_num)
