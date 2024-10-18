import os
import torch
from models import Models

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'PathModel':Models.PathModel
        }
        #self.device = self._acquire_device()
        self.model = self._build_model().cuda()

    def _build_model(self):
        raise NotImplementedError
        return None

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
