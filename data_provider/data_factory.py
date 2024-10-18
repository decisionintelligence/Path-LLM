from data_provider.data_loader import Basic_Dataset,PimDataset
from torch.utils.data import DataLoader
from utils.tools import basic_collate_fn,pim_test_collate_fn,gpt_prompt_collate_fn
from torch.utils.data.dataloader import default_collate
data_dict = {
    'PathLLM':Basic_Dataset
}
collate_fn_dict = {
    'PathLLM':basic_collate_fn
}

def data_provider(args, flag,st=0,end=0):
    Data = data_dict[args.model]
    if flag == 'test':
        shuffle_flag = False
        batch_size = args.batch_size 
    else:
        shuffle_flag = False
        batch_size = args.batch_size  # bsz for train and valid
    
    data_set = Data(
        args,
        flag=flag
    )
    print(flag, len(data_set))
    my_collate_fn = collate_fn_dict[args.model]
    data_loader = DataLoader(
    data_set,
    batch_size=batch_size,
    shuffle=shuffle_flag,collate_fn = my_collate_fn)
    if st == 0 and end == 0:
        pass
    else:
        data_set = Data(
        args,
        flag=flag,
        st=st,end=end
        )
        
        data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,collate_fn = my_collate_fn)
    return data_set, data_loader
