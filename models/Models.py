
import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from layers.layers import TPFModel,PathEncoder
from layers.layers import MaskedAutoencoderViT as MAE
from utils.tools import currilum_dropout
import numpy as np
import torch.nn.functional as F
class PathModel(nn.Module):
    
    def __init__(self, configs,text_embeddings,prompt_embedding,road_embeddings):
        super(PathModel, self).__init__()
        self.batch_size = configs.batch_size
        self.road_size = configs.road_size
        self.out_dim = configs.out_dim
        self.embedding_dim = configs.embedding_dim  #embed_dim
        # self.max_len = configs.max_len

        self.text_embeddings = text_embeddings.float()
        # print(self.text_embeddings.size())
        # self.prompt_embedding = prompt_embedding
        self.road_embeddings = nn.Embedding.from_pretrained(road_embeddings)
        self.road_embeddings.weight.requires_grad = False
        self.text_embeddings_len = text_embeddings.size(1)
        self.mlp = nn.Linear(768,self.out_dim)
        self.mlp1 = nn.Linear(self.text_embeddings_len,self.out_dim)
        self.pathencoder = PathEncoder(768,768)
        self.pathencoder.load_state_dict(torch.load('./xian_dataset/pathencoder3.pth'))
        self.w_text = nn.Parameter(torch.ones(1, 38) / 38).float()
        self.gf = TPFModel(768,768,768)
        self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.005)
        self.ln_proj1 = nn.LayerNorm(self.out_dim)
        self.linear = nn.Linear(self.out_dim,1)
        self.max_pool = nn.MaxPool1d(kernel_size=4, stride=4)

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True           
            else:
                param.requires_grad = False

        # for layer in (self.gpt2,self.road_embeddings,self.text_embeddings,self.adapter1,self.adapter2,self.mlp1,self.linear,self.ln_proj):
        #     # layer.to(device)
        #     layer.train()
    
    def forward(self,x,mask,valid_len,dp = 0):

        mask_3d = torch.unsqueeze(mask, 2).repeat(1, 1, self.out_dim)
        road_embed = self.road_embeddings(x) #batch_size*max_len*embed_dim  
        text_embed = self.text_embeddings[x]

        gpt_input = self.tpf(road_embed,text_embed) 

        gpt_input = self.leakyrelu(gpt_input)
        
        outputs = self.gpt2(inputs_embeds=gpt_input,attention_mask=mask).last_hidden_state  #batch_size*max_len*embed_dim
        outputs = self.leakyrelu(outputs)
        outputs = outputs * mask_3d
        outputs = torch.mean(outputs,1)
        outputs_norm = self.ln_proj1(outputs)
        pred = self.linear(outputs_norm)
        pred = torch.squeeze(pred)
        return pred
    def FFN(self,text_embed,road_embed):
        
        text_embed = self.max_pool(text_embed)
        text_embed = self.leakyrelu(text_embed)
        road_embed = self.mlp(road_embed) 
        road_embed = self.leakyrelu(road_embed)

        road_inputs = torch.cat([road_embed,text_embed],dim=2)
        road_inputs = self.mlp2(road_inputs)
        return road_inputs
        # return road_inputs
    def tpf(self,road_embed,text_embed):
        text_embed = self.max_pool(text_embed)
        # text_embed = self.leakyrelu(text_embed)
        # road_embed = self.mlp(road_embed) 
        # road_embed = self.leakyrelu(road_embed)

        road_embed = self.pathencoder(road_embed)
        road_embed = self.leakyrelu(road_embed)
        text_embed = nn.functional.normalize(text_embed, dim=1)
        road_embed = nn.functional.normalize(road_embed, dim=1)

        fusion_embed = self.gf(road_embed,text_embed)
        fusion_embed = self.ln_proj1(fusion_embed)
        return fusion_embed
    
    
    
    