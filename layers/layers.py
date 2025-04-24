import torch
import torch.nn as nn
from math import sqrt
import math
from functools import partial
from timm.models.vision_transformer import Block
import torch.nn.functional as F
import random
from utils.tools import trunc_normal_
class PathEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PathEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)
class Discriminator(nn.Module):
    def __init__(self, n_h1, n_h2):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h1, n_h2, 1)
        self.act = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, h_c, h_ori, h_p, h_n1, h_n2, s_bias1=None, s_bias2=None):

        ##Positve samples
        h_c = h_c.expand_as(h_ori).contiguous()
        h_p = h_p.expand_as(h_ori).contiguous()

        sc_p1 = torch.squeeze(self.f_k(h_p, h_c), 2)

        ##Negative samples
        h_n1 = h_n1.expand_as(h_ori).contiguous()
        h_n2 = h_n2.expand_as(h_ori).contiguous()


        sc_n1 = torch.squeeze(self.f_k(h_n1, h_c), 2)
        sc_n2 = torch.squeeze(self.f_k(h_n2, h_c), 2)

        logits = torch.cat((sc_p1, sc_n1, sc_n2), 1)

        return logits
class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            return torch.sum(seq * msk, 1) / torch.sum(msk,1) 
class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size,
                 n_layers=2, dropout=0.5):
        super(EncoderLSTM, self).__init__()

        ##can be initialized by results from graph embedding methods, e.g. node2vec.
        self.embedding= nn.Embedding(4316 , 768)
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout)

    def forward(self, path, hidden=None):

        PathEmbedOri = self.embedding(path)
        PathEmbed = PathEmbedOri.transpose(1,0)

        outputs, hidden = self.lstm(PathEmbed, hidden)
        outputs = outputs.transpose(0,1)
        
        return outputs, hidden, PathEmbedOri
class PIM(nn.Module):
    def __init__(self, n_in, n_h, input_size=768, hidden_size= 768, n_layers=1, dropout=0.5):
        
        super(PIM, self).__init__()

        self.encoder_path = EncoderLSTM(input_size=input_size, hidden_size=hidden_size, n_layers=n_layers, dropout=dropout)
        
        
        self.disc_node = Discriminator(n_in, n_h)
        self.disc_path = Discriminator(n_in, n_h)
        
        self.read = self.read = Readout()

        #can be initialized by results from graph embedding methods, e.g. node2vec.
        self.embeddingn= nn.Embedding(4316,768)


    def forward(self, Path1, Path1Mask, Path2, Path2Mask, Path3, Path3Mask, Pos1, Pos1Mask, Neg1, Neg1Mask, Neg2, Neg2Mask):


        ###path 
        Path1Out, _, Path1Ori = self.encoder_path(Path1)
        Path1Embed = torch.unsqueeze(self.read(Path1Out, Path1Mask),1)
        Path1OriP = torch.unsqueeze(self.read(Path1Ori, Path1Mask),1)

        Path2Out, _, _ = self.encoder_path(Path2)
        Path2Embed = torch.unsqueeze(self.read(Path2Out, Path2Mask),1) 
    

        Path3Out, _, _ = self.encoder_path(Path3)
        Path3Embed = torch.unsqueeze(self.read(Path3Out, Path3Mask),1)

        ###Node embedding
        NEmbedPos = self.embeddingn(Pos1)
        NEmbedNeg1 = self.embeddingn(Neg1)
        NEmbedNeg2= self.embeddingn(Neg2)

        ##node_discriminator
        logits1 = self.disc_node(Path1Embed, Path1Ori, NEmbedPos*Pos1Mask, NEmbedNeg1*Neg1Mask, NEmbedNeg2*Neg2Mask)  
        
        ##path_discriminator
        logits2 = self.disc_path(Path1Embed, Path1OriP, Path1OriP, Path2Embed, Path3Embed)
        
        
        return logits1, logits2

    def embed(self, Path1, Path1Mask):
        
        #path embedding
        Path1Out, _, _ = self.encoder_path(Path1)
        # print(Path1Out.size())
        # Path1Embed = torch.unsqueeze(self.read(Path1Out, Path1Mask),1)
        # return Path1Embed.detach()
        return Path1Out
class PatchEmbed(nn.Module):
    def __init__(self, node2vec, embed_dim=768, dropout=0.1):
        super().__init__()
        # self.token = nn.TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        # self.token = nn.Embedding(90000, 128)
        self.token = nn.Embedding.from_pretrained(node2vec)
        #self.time = nn.Embedding.from_pretrained(time2vec)
        # self.token = nn.Embedding.from_pretrained(node2vec,freeze=False) 
        # self.time = nn.Embedding.from_pretrained(time2vec,freeze=False)
        # self.rt = nn.Embedding.from_pretrained(roadtype2vec)
        # self.ow = nn.Embedding.from_pretrained(oneway2vec)
        # self.lane = nn.Embedding.from_pretrained(lane2vec)
        # self.rt = nn.Embedding(21, 64)
        # self.ow = nn.Embedding(3,16)
        # self.lane = nn.Embedding(7,32)
        # self.dropout = nn.Dropout(p=dropout)
        # self.norm = nn.LayerNorm(128)
        # self.fc1 = nn.Linear(128,512)
        # self.fc2 = nn.Linear(128,512)


    def forward(self, seq):#, seq_rt, seq_ow, seq_lane):
        # x = self.fc1(torch.cat([self.token(seq), self.time(ts)],dim=2))
        # x = self.fc1(self.token(seq))
        x = self.token(seq)
        #ts_ =self.time(ts)

        # ts_ = self.fc2(self.time(ts))
        # x = torch.cat([self.token(seq), self.rt(seq_rt),self.ow(seq_ow),self.lane(seq_lane)],dim=2)
        return x #self.norm(x) ####BNC where c =1
    

#LightPath
class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, 
                 node2vec,
                 # moving_average_decay,
                 # beta,
                 # projection_size,
                 # projection_hidden_size,
                 # prediction_hidden_size,
                 # prediction_size,
                 num_patches=515, in_chans=3,
                 embed_dim=768, depth=24, num_heads=16,
                 decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., embed_layer=PatchEmbed, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = embed_layer(node2vec=node2vec,embed_dim=768, dropout=0.)

        num_patches = num_patches
        self.num_classes=3

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #print(f'----------{num_patches}------------')
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.relation_head = torch.nn.Sequential(
                             torch.nn.Linear(768*2, 768),  ###change 2 to 4
                             torch.nn.BatchNorm1d(768),
                             torch.nn.LeakyReLU(),
                             torch.nn.Linear(768, 1))
        self.norm_pix_loss = norm_pix_loss
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.criterion_cls = nn.CrossEntropyLoss()

        self.clstoken =nn.Linear(768*2,3)
        self.acf = nn.ReLU()

        self.apply(self._init_vit_weights)

    def _init_vit_weights(self, module):
        """ ViT weight initialization
        """
        if isinstance(module, nn.Linear):
            if module.out_features == self.num_classes:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)             
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x,mask_ratio):
        # embed patches
        #print(x.size())
        x = self.patch_embed(x)
        xx=x
        #(x.size())

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore,xx

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        # x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, x, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # target = self.patchify(imgs)
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - x) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, intputs, mask_ratio1=0.7, mask_ratio2=0.8):
        
        latent1, mask1, ids_restore1, xx1 = self.forward_encoder(intputs,mask_ratio1)
        latent2, mask2, ids_restore2, xx2 = self.forward_encoder(intputs,mask_ratio2)
        
        pred1 = self.forward_decoder(latent1, ids_restore1)  # [N, L, p*p*3]
        pred2 = self.forward_decoder(latent2, ids_restore2)  # [N, L, p*p*3]
        
        cls_token_1 = latent1[:, 0, :]
        cls_token_2 = latent2[:, 0, :]

        # cls_token_rr1 = torch.cat([cls_token_1, torch.squeeze(ts1[:,0,:],1)],1)
        # cls_token_rr2 = torch.cat([cls_token_2, torch.squeeze(ts2[:,0,:],1)],1)
        # #####CLS
        #cls_token_c1 = self.acf(self.clstoken(torch.cat([cls_token_1, torch.squeeze(ts1[:,0,:],1)],1)))
        #cls_token_c2 = self.acf(self.clstoken(torch.cat([cls_token_2, torch.squeeze(ts2[:,0,:],1)],1)))

        # ###peak offpeak classification
        # loss_cls1 =self.criterion_cls(torch.squeeze(cls_token_c1, 1),targets)
        # loss_cls2 =self.criterion_cls(torch.squeeze(cls_token_c2, 1),targets)
        # loss_cls = 0.5*(loss_cls1 + loss_cls2)

        ####Rec
        loss_rec1 = self.forward_loss(xx1, pred1, mask1)
        loss_rec2 = self.forward_loss(xx2, pred2, mask2)
        loss_rec = 0.5*(loss_rec1 + loss_rec2)

        #####RR
        
        cls_feature_rr = torch.cat([cls_token_1, cls_token_2],0) 
        relation_pairs, targets_rr, scores = self.aggregate(cls_feature_rr, K=2)
        if torch.cuda.is_available():
            scores = scores.cuda()
            targets_rr = targets_rr.cuda()
        loss_rr   =self.criterion_bce(torch.squeeze(scores,1), targets_rr)

        return loss_rec, loss_rec, loss_rr, pred1, mask1, pred2, mask2

    def aggregate(self, features, K=2):
        relation_pairs_list = list()
        targets_list = list()
        size = int(features.shape[0] / K)
        shifts_counter=1
        for index_1 in range(0, size*K, size):
            for index_2 in range(index_1+size, size*K, size):
                # Using the 'cat' aggregation function by default
                pos_pair = torch.cat([features[index_1:index_1+size], 
                              features[index_2:index_2+size]], 1)
                # Shuffle without collisions by rolling the mini-batch (negatives)
                neg_pair = torch.cat([
                     features[index_1:index_1+size], 
                     torch.roll(features[index_2:index_2+size], 
                     shifts=shifts_counter, dims=0)], 1)
                relation_pairs_list.append(pos_pair)
                relation_pairs_list.append(neg_pair)
                targets_list.append(torch.ones(size, dtype=torch.float32))
                targets_list.append(torch.zeros(size, dtype=torch.float32))
                shifts_counter+=1
                if(shifts_counter>=size): 
                    shifts_counter=1 # avoid identity pairs
        relation_pairs = torch.cat(relation_pairs_list, 0)
        targets = torch.cat(targets_list, 0)
        # print(relation_pairs.size())
        return relation_pairs, targets, self.relation_head(relation_pairs)
    
    def embed(self, intputs,mask_ratio=0.75):
        latent, _,_,_,= self.forward_encoder(intputs,mask_ratio)
        return latent#[:,0,:]
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
class Adapter(nn.Module):

    def __init__(self,embedding_dim,hidden_dim):
        super(Adapter, self).__init__()
        self.Wd = nn.Parameter(torch.randn(embedding_dim,hidden_dim))
        self.Wu = nn.Parameter(torch.randn(hidden_dim,768))
        # self.mlp = nn.Linear(embedding_dim,embedding_dim)
        self.relu = nn.ReLU()
    def forward(self,X):
        # X = self.mlp(X)
        intermediate = self.relu(torch.matmul(X,self.Wd))
        output = torch.matmul(intermediate,self.Wu)
        
        return output

class CoAttention(nn.Module):
    def __init__(self, input_dim):
        super(CoAttention, self).__init__()
        self.W1 = nn.Linear(input_dim, input_dim)
        self.W2 = nn.Linear(input_dim, input_dim)
        self.W3 = nn.Linear(input_dim*2, input_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.leakyrelu = nn.LeakyReLU()
    def forward(self, input1, input2):
        # input1: [batch_size, seq_len1, input_dim]
        # input2: [batch_size, seq_len2, input_dim]

        # Compute attention weights
        attention_weights1 = self.softmax(torch.matmul(input1, self.W1(input2).transpose(1, 2)))
        attention_weights2 = self.softmax(torch.matmul(input2, self.W2(input1).transpose(1, 2)))

        # Attend to input sequences
        attended_input1 = torch.matmul(attention_weights1, input2)
        attended_input2 = torch.matmul(attention_weights2, input1)

        # Fusion
        fusion1 = self.leakyrelu(self.W3(torch.cat((input1, attended_input1), dim=-1)))
        fusion2 = self.leakyrelu(self.W3(torch.cat((input2, attended_input2), dim=-1)))

        return fusion1, fusion2
class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size * 1, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x, mask=None):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * 1, x.size(0), self.hidden_size).to(x.device)
        h0 = self.relu(h0)
        # Bidirectional GRU
        out, _ = self.gru(x, h0)
        out = self.relu(out)
        # Apply mask if provided
        if mask is not None:
            out = out * mask.unsqueeze(-1)  # Broadcasting mask to match out's shape

        # Sum along the sequence dimension if mask is provided
        
        out = out.sum(dim=1)
        # Fully connected layer
        out = self.fc(out)
        return out
    
class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)
        self.logsoftmax = nn.LogSoftmax(dim = -1)
        
    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(self.logsoftmax(scale * scores))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
    
# class GatedFusion(nn.Module):
#     def __init__(self, embedding_dim):
#         super(GatedFusion, self).__init__()
#         self.P_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
#         self.P_a = nn.Linear(embedding_dim, embedding_dim, bias=False)
#         self.bg = nn.Parameter(torch.zeros(embedding_dim))
#         # self.leakyrelu = nn.LeakyReLU()

#     def forward(self, C_v, C_a):
#         # C_v and C_a are tensors with shape (batch_size, len, embedding_dim)
#         P_vC_v = self.P_v(C_v)  # Apply linear transformation to C_v
#         # P_vC_v = self.leakyrelu(P_vC_v)
#         P_aC_a = self.P_a(C_a)  # Apply linear transformation to C_a
#         # P_aC_a = self.leakyrelu(P_aC_a)

#         # Compute gating coefficient alpha
#         alpha = torch.sigmoid(P_vC_v + P_aC_a + self.bg)
    
#         # Compute joint representation J_va
#         J_va = alpha * C_a + (1 - alpha) * C_v
#         # J_va = torch.cat([alpha * C_a , (1 - alpha) * C_v],dim = 2)  
#         return J_va
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, key_size, value_size, bias=False):
        super(MultiHeadSelfAttention,self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_head_dim = key_size // num_heads
        self.k_head_dim = key_size // num_heads
        self.v_head_dim = value_size // num_heads
        
        self.W_q = nn.Linear(embed_dim, key_size, bias=bias)
        self.W_k = nn.Linear(embed_dim, key_size, bias=bias)
        self.W_v = nn.Linear(embed_dim, value_size, bias=bias)        

        self.q_proj = nn.Linear(key_size, key_size, bias=bias)
        self.k_proj = nn.Linear(key_size, key_size, bias=bias)
        self.v_proj = nn.Linear(value_size, value_size, bias=bias)
        self.out_proj = nn.Linear(value_size, embed_dim, bias=bias)

    def forward(self, x):

        query = self.W_q(x)  # (N, L, key_size)
        key = self.W_k(x)  # (N, L, key_size)
        value = self.W_v(x)  # (N, L, value_size)
        q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(value)
        N, value_size = v.size()

        q = q.reshape(N,self.num_heads, self.q_head_dim).transpose(1, 2)
        k = k.reshape(N,self.num_heads, self.k_head_dim).transpose(1, 2)
        v = v.reshape(N,self.num_heads, self.v_head_dim).transpose(1, 2)

        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(k.size(-1))
        att = F.softmax(att, dim=-1)
        output = torch.matmul(att, v)
        output = output.transpose(1, 2).reshape(N,value_size)
        output = self.out_proj(output)
        
        return output
class TPFModel(nn.Module):
    def __init__(self, num_hidden_a, num_hidden_b, num_hidden):
        super(TPFModel, self).__init__()
        self.hidden = num_hidden
        self.w1 = nn.Parameter(torch.Tensor(num_hidden_a, num_hidden))
        self.w2 = nn.Parameter(torch.Tensor(num_hidden_b, num_hidden))
        self.bias = nn.Parameter(torch.Tensor(num_hidden))
        self.reset_parameter()

    def reset_parameter(self):
        stdv1 = 1. / math.sqrt(self.hidden)
        stdv2 = 1. / math.sqrt(self.hidden)
        stdv = (stdv1 + stdv2) / 2.
        self.w1.data.uniform_(-stdv1, stdv1)
        self.w2.data.uniform_(-stdv2, stdv2)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, a, b):
        wa = torch.matmul(a, self.w1)
        wb = torch.matmul(b, self.w2)
        gated = wa + wb + self.bias
        gate = torch.sigmoid(gated)
        # print(gate.size(),a.size())
        output = gate * a + (1 - gate) * b
        return output  # Clone the tensor to make it out of place operation
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # Batch first to sequence first
        attn_output, _ = self.attention(x, x, x)
        return attn_output.permute(1, 0, 2)  # Sequence first back to batch first

class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            return torch.sum(seq * msk, 1) / torch.sum(msk,1)   
class Local_Node(nn.Module):


    def __init__(self):
        super(Local_Node, self).__init__()
      
    def forward(self, h_c, h_p1, h_p2, h_n1, h_n2, h_n3):

        num_node = 10
        seq_len = h_p1.size(1)-1
        loss_node1 =0
        loss_node2 =0
        for _ in range(num_node):
            node_index = random.randint(1,seq_len)
     
            h_p1_ = h_p1[:,node_index,:]
       
    
            h_n1_ = h_n1[:,node_index,:]
            h_n2_ = h_n2[:,node_index,:]
            h_n3_ = h_n3[:,node_index,:]
            loss_node1 += self.contrastive_loss_sim(h_c, h_p1_, h_n1_, h_n2_, h_n3_)
        
        for _ in range(num_node):
            node_index = random.randint(1,seq_len)
            h_p2_ = h_p2[:,node_index,:]
       
    
            h_n1_ = h_n1[:,node_index,:]
            h_n2_ = h_n2[:,node_index,:]
            h_n3_ = h_n3[:,node_index,:]
            loss_node2 += self.contrastive_loss_sim(h_c, h_p2_, h_n1_, h_n2_, h_n3_)
     
        loss_node_ = 1/(2*num_node)*(loss_node1+loss_node2)
    def contrastive_loss_sim(self,hc, hp1, hn1, hn2, hn3):
        f = lambda x: torch.exp(x)
        p_sim = f(self.sim(hc, hp1))
        n_sim1 = f(self.sim(hc, hn1))
        n_sim2 = f(self.sim(hc, hn2))
        n_sim3 = f(self.sim(hc, hn3))
        return -torch.log(p_sim.diag() /
                          (p_sim.sum(dim=-1) + n_sim1.sum(dim=-1) - n_sim1.diag() +
                           n_sim2.sum(dim=-1) - n_sim2.diag() + n_sim3.sum(dim=-1) - n_sim3.diag()))

    def sim(h1, h2):
        z1 = F.normalize(h1, dim=-1, p=2)
        z2 = F.normalize(h2, dim=-1, p=2)
        return torch.mm(z1, z2.t())
