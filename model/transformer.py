import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from util import clones
from transformers.activations import get_activation

def self_attention(query, key, value, mask=None):
    key_transpose = torch.transpose(key,-2,-1)                      # (bath, head_num, d_k, token_)
    matmul_result = torch.matmul(query,key_transpose)                # MatMul(Q,K)
    d_k = query.size()[-1]
    attention_score = matmul_result/math.sqrt(d_k)                  # Scale

    if mask is not None:
        attention_score = attention_score.masked_fill(mask == 0, -1e20)

    softmax_attention_score = F.softmax(attention_score,dim=-1)  # attention value
    result = torch.matmul(softmax_attention_score,value)

    return result, softmax_attention_score


class MultiHeadAttention(nn.Module):
    def __init__(self, head_num =8 , d_model = 300,dropout = 0.1):
        super(MultiHeadAttention,self).__init__()

        # print(d_model % head_num)
        # assert d_model % head_num != 0

        self.head_num = head_num
        self.d_model = d_model
        self.d_k = self.d_v = d_model // head_num

        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)

        self.self_attention = self_attention
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask = None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        batche_num = query.size(0)

        query = self.w_q(query).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)

        attention_result, attention_score = self.self_attention(query, key, value, mask)

        attention_result = attention_result.transpose(1,2).contiguous().view(batche_num, -1, self.head_num * self.d_k)


        return self.w_o(attention_result)

class FeedForward(nn.Module):
    def __init__(self,d_model, dropout = 0.1):
        super(FeedForward,self).__init__()
        self.w_1 = nn.Linear(d_model, d_model*4)
        self.w_2 = nn.Linear(d_model*4, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim =True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x-mean)/ (std + self.eps) + self.b_2

class ResidualConnection(nn.Module):
    def __init__(self, size, dropout):
        super(ResidualConnection,self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout((sublayer(self.norm(x))))

class Encoder(nn.Module):
    def __init__(self, d_model, head_num, dropout):
        super(Encoder,self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model= d_model, head_num= head_num)
        self.residual_1 = ResidualConnection(d_model,dropout=dropout)

        self.feed_forward = FeedForward(d_model)
        self.residual_2 = ResidualConnection(d_model,dropout=dropout)

    def forward(self, input, mask):
        x = self.residual_1(input, lambda x: self.multi_head_attention(x, x, x, mask))
        x = self.residual_2(x, lambda x: self.feed_forward(x))
        return x

class Decoder(nn.Module):
    def __init__(self, d_model,head_num, dropout):
        super(Decoder,self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(d_model= d_model, head_num= head_num)
        self.residual_1 = ResidualConnection(d_model,dropout=dropout)

        self.encoder_decoder_attention = MultiHeadAttention(d_model= d_model, head_num= head_num)
        self.residual_2 = ResidualConnection(d_model,dropout=dropout)

        self.feed_forward= FeedForward(d_model)
        self.residual_3 = ResidualConnection(d_model,dropout=dropout)


    def forward(self, target, encoder_output, target_mask, encoder_mask):
        # target, x, target_mask, input_mask
        x = self.residual_1(target, lambda x: self.masked_multi_head_attention(x, x, x, target_mask))
        x = self.residual_2(x, lambda x: self.encoder_decoder_attention(x, encoder_output, encoder_output, encoder_mask))
        x = self.residual_3(x, self.feed_forward)

        return x

class Embeddings(nn.Module):
    def __init__(self, vocab_num, d_model):
        super(Embeddings,self).__init__()
        self.emb = nn.Embedding(vocab_num,d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model,dropout=0.1):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)

        position = torch.arange(0,max_seq_len, dtype=torch.float32).unsqueeze(1)
        base = torch.ones(d_model//2).fill_(10000)
        pow_term = torch.arange(0, d_model, 2, dtype=torch.float32) / torch.tensor(d_model)

        div_term = torch.pow(base,pow_term)

        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('positional_encoding', pe)

    def forward(self, x):
        x = x + Variable(self.positional_encoding[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class PositionEmbedding(nn.Module):
    def __init__(self, max_seq_len, dim):
        super(PositionEmbedding,self).__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos_list):
        pos = torch.tensor(pos_list)
        if torch.cuda.is_available():
            pos = pos.cuda()

        a = self.pos_embedding(pos)
        x = x + self.pos_embedding(pos)
        return x

class Generator(nn.Module):
    def __init__(self, d_model, vocab_num):
        super(Generator, self).__init__()
        self.proj_1 = nn.Linear(d_model, d_model*4)
        self.proj_2 = nn.Linear(d_model*4, vocab_num)

    def forward(self, x):
        x = self.proj_1(x)
        x = self.proj_2(x)
        return x

class Transformer(nn.Module):
    def __init__(self,vocab_num, d_model, max_seq_len, head_num, dropout, N, sync_pos=False):
        super(Transformer,self).__init__()
        self.embedding = Embeddings(vocab_num, d_model)
        self.sync_pos = sync_pos
        if self.sync_pos:
            self.positional_encoding = PositionEmbedding(max_seq_len,d_model)
        else:
            self.positional_encoding = PositionalEncoding(max_seq_len,d_model)

        self.encoders = clones(Encoder(d_model=d_model, head_num=head_num, dropout=dropout), N)
        self.decoders = clones(Decoder(d_model=d_model, head_num=head_num, dropout=dropout), N)

        self.generator = Generator(d_model, vocab_num)

    def forward(self, input, target, input_mask, target_mask, labels=None, pos=None, sync_pos=None):
        if self.sync_pos:
            x = self.positional_encoding(self.embedding(input), pos)
        else:
            x = self.positional_encoding(self.embedding(input))
        input_mask = input_mask.unsqueeze(1)
        target_mask = target_mask.unsqueeze(1)
        for encoder in self.encoders:
            x = encoder(x, input_mask)

        if self.sync_pos:
            target = self.positional_encoding(self.embedding(target), sync_pos)
        else:
            target = self.positional_encoding(self.embedding(target))

        for decoder in self.decoders:
            # target, encoder_output, target_mask, encoder_mask)
            target = decoder(target, x, target_mask, input_mask)

        lm_logits = self.generator(target)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return lm_logits, loss

    def encode(self,input, input_mask):
        if self.sync_pos:
            x = self.positional_encoding(self.embedding(input), pos)
        else:
            x = self.positional_encoding(self.embedding(input))
        for encoder in self.encoders:
            x = encoder(x, input_mask)
        return x

    def decode(self, encode_output, encoder_mask, target, target_mask):
        if self.sync_pos:
            target = self.positional_encoding(self.embedding(target), sync_pos)
        else:
            target = self.positional_encoding(self.embedding(target))
        for decoder in self.decoders:
            #target, encoder_output, target_mask, encoder_mask
            target = decoder(target, encode_output, target_mask, encoder_mask)

        lm_logits = self.generator(target)

        return lm_logits
