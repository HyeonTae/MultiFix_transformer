import os
import torch
from torch.autograd import Variable
from util import subsequent_mask
from transformer import Transformer
from transformers import BertTokenizer
import argparse
par = argparse.ArgumentParser()
par.add_argument("-sp", "--sync_pos", action='store_true')
args = par.parse_args()

if __name__=="__main__":
    sync_pos = args.sync_pos
    print("Start predict whit synchronized position" if args.sync_pos else "Start predict")

    data_name = 'DeepFix'
    project_dir = '../..'
    #vocab_path = '../vocab/vocab.model'
    vocab_path = '../vocab/vocab.txt'
    #data_path = '../data/' + data_name + '/train.csv'
    model_name = 'transformer_' + data_name + ('_sync_pos' if sync_pos else '')
    model_path = '../log/pth/' + model_name + '_model_save.pth'

    if sync_pos:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load('../vocab/vocab.model')
        insert_tok = list()
        for i in range(1,422):
            insert_tok.append(sp.piece_to_id('▁' + str(i)))

    # model setting
    vocab_num = 1270
    max_length = 400
    d_model = 200
    head_num = 8
    dropout = 0.1
    N = 6
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = BertTokenizer(vocab_path, do_lower_case=False, do_basic_tokenize=False)
    model = Transformer(vocab_num=vocab_num,
                      d_model=d_model,
                      max_seq_len=max_length,
                      head_num=head_num,
                      dropout=dropout,
                      N=N,
                      sync_pos=sync_pos)
    
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    while True:
        input_str = "_<directive>_#include _<include>_<stdio.h> _<type>_int _<APIcall>_main _<op>_( _<op>_) _<op>_{ _<type>_int _<id>_1@ _<op>_, _<id>_5@ _<op>_, _<id>_2@ _<op>_= _<number>_# _<op>_; _<APIcall>_scanf _<op>_( _<string>_ _<op>_, _<op>_& _<id>_1@ _<op>_, _<op>_& _<id>_5@ _<op>_) _<op>_; _<type>_int _<id>_4@ _<op>_[ _<id>_1@ _<op>_] _<op>_; _<keyword>_for _<op>_( _<type>_int _<id>_6@ _<op>_= _<number>_# _<op>_; _<id>_6@ _<op>_< _<id>_1@ _<op>_; _<id>_6@ _<op>_++ _<op>_) _<APIcall>_scanf _<op>_( _<string>_ _<op>_, _<op>_& _<id>_4@ _<op>_[ _<id>_6@ _<op>_] _<op>_) _<op>_; _<keyword>_for _<op>_( _<id>_3@ _<op>_= _<number>_# _<op>_; _<id>_3@ _<op>_< _<id>_5@ _<op>_; _<id>_3@ _<op>_++ _<op>_) _<op>_{ _<keyword>_for _<op>_( _<type>_int _<id>_6@ _<op>_= _<number>_# _<op>_; _<id>_6@ _<op>_< _<id>_1@ _<op>_; _<id>_6@ _<op>_++ _<op>_) _<op>_{ _<keyword>_if _<op>_( _<id>_4@ _<op>_[ _<id>_6@ _<op>_] _<op>_> _<id>_2@ _<op>_) _<id>_2@ _<op>_= _<id>_4@ _<op>_[ _<id>_6@ _<op>_] _<op>_; _<op>_} _<APIcall>_printf _<op>_( _<string>_ _<op>_, _<id>_2@ _<op>_) _<op>_; _<keyword>_for _<op>_( _<type>_int _<id>_6@ _<op>_= _<number>_# _<op>_; _<id>_6@ _<op>_< _<id>_1@ _<op>_; _<id>_6@ _<op>_++ _<op>_) _<op>_{ _<keyword>_if _<op>_( _<id>_4@ _<op>_[ _<id>_6@ _<op>_] _<op>_= _<id>_2@ _<op>_) _<id>_4@ _<op>_[ _<id>_6@ _<op>_] _<op>_= _<number>_# _<op>_; _<op>_} _<op>_} _<keyword>_return _<number>_# _<op>_; _<op>_}"
        str = tokenizer.encode_plus(input_str, return_tensors='pt',
                        padding='max_length', truncation=True, max_length=400)

        print(str["input_ids"])

        encoder_input = str["input_ids"].to(device)
        encoder_mask = str["attention_mask"].to(device)

        print("pad: {}".format(tokenizer.pad_token_id))
        print("cls: {}".format(tokenizer.cls_token_id))
        print("sep: {}".format(tokenizer.sep_token_id))
        print("unk: {}".format(tokenizer.unk_token_id))

        input_length = len(input_str.split())
        print("input_length: {}".format(input_length))
        pos = []
        for m in range(max_length):
            if input_length - m > 0:
                pos.append(input_length - m)
            else:
                pos.append(0)

        input_pos = torch.tensor(pos).unsqueeze(0).to(device)

        print("decode: {}".format(tokenizer.batch_decode(str["input_ids"])))

        #pad_len = (max_length - len(str))
        #str_len = len(str)
        #encoder_input = torch.tensor(str + [tokenizer.pad_token_id]* pad_len)
        #encoder_masks = (encoder_input != tokenizer.pad_token_id).unsqueeze(0)

        #target = torch.ones(1, 1).fill_(tokenizer.unk_token_id).type_as(encoder_input)

        ###qq=tokenizer.encode_plus('<bos>', return_tensors='pt',
        ###                padding='max_length', truncation=True, max_length=400)
        ###print(qq)

        target = torch.zeros(1, 1).fill_(tokenizer.cls_token_id).type_as(encoder_input).to(device)
        #target = torch.zeros(1, 1).fill_(tokenizer.pad_token_id).type_as(encoder_input)
        print(target)
        print(Variable(subsequent_mask(target.size(1)).type_as(encoder_input)))

        if sync_pos:
            encoder_output = model.encode(encoder_input, encoder_mask.unsqueeze(1), input_pos)
        else:
            encoder_output = model.encode(encoder_input, encoder_mask.unsqueeze(1))

        syncpos = [[input_length]]
        for i in range(max_length - 1):
            target_syncpos = torch.tensor(syncpos).to(device)
            if sync_pos:
                lm_logits = model.decode(encoder_output, encoder_mask, target,
                        Variable(subsequent_mask(target.size(1)).type_as(encoder_input)).to(device)
                        , target_syncpos)
            else:
                lm_logits = model.decode(encoder_output, encoder_mask, target,
                        Variable(subsequent_mask(target.size(1)).type_as(encoder_input)).to(device))
            prob = lm_logits[:, -1]
            _, next_word = torch.max(prob, dim=1)

            if next_word.data[0] == tokenizer.pad_token_id or next_word.data[0] == tokenizer.sep_token_id:
                print(f'token: {input_str}\ntarget: {tokenizer.decode(target.squeeze().tolist(),skip_special_tokens=True)}')
                break
            target = torch.cat((target[0], next_word))
            if sync_pos:
                if input_length <= 0:
                    syncpos[0].append(0)
                else:
                    if target[i] not in insert_tok:
                        input_length -= 1
                    syncpos[0].append(input_length)
            target = target.unsqueeze(0)
            #print(syncpos)

        print(f'token: {input_str}\ntarget: {tokenizer.decode(target.squeeze().tolist())}')
        print(target.shape)
        break
