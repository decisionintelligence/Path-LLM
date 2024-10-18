import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
import matplotlib.pyplot as plt
from einops import repeat
import warnings
import math
plt.switch_backend('agg')
def currilum_dropout(epoch,p_min):
        gamma = 0.1
        p = ( - p_min) * math.exp(-gamma * epoch) + p_min
        
        return p
def pim_train_collate_fn(batch):
    
    max_len_batch = max([i[12] for i in batch])
    batch = [(i[0][:max_len_batch],i[1][:max_len_batch],i[2][:max_len_batch],i[3][:max_len_batch],i[4][:max_len_batch]
              ,i[5][:max_len_batch],i[6][:max_len_batch],i[7][:max_len_batch],i[8][:max_len_batch],i[9][:max_len_batch],i[10][:max_len_batch],i[11][:max_len_batch],i[12]) for i in batch]
    # print(max_len_batch)
    
    return default_collate(batch)

def basic_collate_fn(batch):

    max_len_batch = max([i[2] for i in batch])
    batch = [(i[0][:max_len_batch],i[1],i[2]) for i in batch]
    # print(max_len_batch)
    # print(batch)
    return default_collate(batch)
def gpt_prompt_collate_fn(batch):

    max_len_batch = max([i[2] for i in batch])
    batch = [(i[0][:max_len_batch],i[1],i[2],i[3]) for i in batch]
    # print(max_len_batch)
    # print(batch)
    return default_collate(batch)
def pim_test_collate_fn(batch):
    
    max_len_batch = int(max([i[3] for i in batch]))
    
    batch = [(i[0][:max_len_batch],i[1],i[2][:max_len_batch],i[3]) for i in batch]
    # print(max_len_batch)
    # print(batch)
    return default_collate(batch)

def get_batch_mask(B,L,valid_len):
    mask = repeat(torch.arange(end=L,device=valid_len.device),'L -> B L',B=B) < repeat(valid_len,'B -> B L',L=L)
    return mask
def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    if args.lradj == 'type7':
        lr_adjust = {epoch: args.learning_rate * (0.7 ** ((epoch - 1) // 1))}
    if args.lradj == 'type6':
        lr_adjust = {epoch: args.learning_rate * (0.6 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class  EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred

def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
