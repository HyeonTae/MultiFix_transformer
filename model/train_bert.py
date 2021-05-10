import torch
from torch.autograd import Variable
from transformer import Transformer
from transformers import BertTokenizer
from tqdm import tqdm
from util import subsequent_mask
import pandas as pd
import math
import os
import time
import argparse
import os

par = argparse.ArgumentParser()
par.add_argument("-sp", "--sync_pos", action='store_true')
args = par.parse_args()

class Trainer():
    def __init__(self,
                 tokenizer,
                 model,
                 max_len,
                 device,
                 model_name,
                 batch_size,
                 use_sync_pos,
                 ):
        self.tokenizer = tokenizer
        self.model = model
        self.max_len = max_len
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.use_sync_pos = use_sync_pos
        self.pad_token_idx = tokenizer.pad_token_id

    def make_std_mask(self, tgt, pad_token_idx):
        'Create a mask to hide padding and future words.'
        target_mask = (tgt != pad_token_idx).unsqueeze(-2)
        target_mask = target_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(target_mask.data))
        return target_mask.squeeze()

    def set_data(self, train_path, val_path):
        train_df = pd.read_csv(train_path, index_col=[0])
        train_df = train_df.sample(frac = 1)

        val_df = pd.read_csv(val_path, index_col=[0])
        val_df = val_df.sample(frac = 1)

        return train_df, val_df

    def train(self, epochs, train_dataset, eval_dataset, optimizer, scheduler):
        self.model.train()
        total_loss = 0.
        global_steps = 0
        start_time = time.time()
        losses = {}
        best_val_loss = float("inf")
        best_model = None
        start_epoch = 0
        start_step = 0
        train_dataset_length = len(train_dataset)

        self.model.to(self.device)
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            train_num_of_batches = int(train_dataset_length/self.batch_size)
            pb = tqdm(range(train_num_of_batches),
                      desc=f'Epoch-{epoch} Iterator',
                      total=train_num_of_batches,
                      bar_format='{l_bar}{bar:10}{r_bar}'
                      )
            pb.update(start_step)

            for i in pb:
                if i < start_step:
                    continue

                input_list=[]
                target_list=[]
                if self.use_sync_pos:
                    pos_list=[]
                    sync_pos_list=[]
                else:
                    pos_list=None
                    sync_pos_list=None
                data = train_dataset[i*batch_size:i*batch_size+batch_size]

                targets_mask = torch.Tensor()
                for indx, row in data.iterrows():
                    _input = row['token']
                    _target = row['target']

                    input_list.append(_input)
                    target_list.append(_target)
                    if self.use_sync_pos:
                        pos_list.append(list(map(int,row['pos'].split())))
                        sync_pos_list.append(list(map(int,row['sync_pos'].split())))
                inputs = tokenizer.batch_encode_plus(input_list, return_tensors='pt',
                        padding='max_length', truncation=True, max_length=400)
                targets = tokenizer.batch_encode_plus(target_list, return_tensors='pt',
                        padding='max_length', truncation=True, max_length=400)

                #print('inputs.shape: {}'.format(inputs["input_ids"].shape))
                #print('decode: {}'.format(tokenizer.decode(inputs["input_ids"][0])))
                #print('decode: {}'.format(tokenizer.decode(targets["input_ids"][0])))

                for tg in targets["input_ids"]:
                    targets_mask = torch.cat((targets_mask,
                        self.make_std_mask(tg, self.pad_token_idx).unsqueeze(0)),dim=0)
                inputs_mask = inputs["attention_mask"].unsqueeze(1)

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                generator_logit, loss = self.model.forward(inputs["input_ids"], targets["input_ids"],
                        inputs_mask.to(self.device), targets_mask.to(self.device), labels=targets["input_ids"],
                        pos=pos_list, sync_pos=sync_pos_list)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                losses[global_steps] = loss.item()
                total_loss += loss.item()
                log_interval = 1
                save_interval = 5000

                global_steps += 1

                if i % log_interval == 0 and i > 0:
                    cur_loss = total_loss / log_interval
                    elapsed = time.time() - start_time
                    pb.set_postfix_str('| epoch {:3d} | {:5d}/{:5d} batches | '
                                     'lr {:02.2f} | ms/batch {:5.2f} | '
                                     'loss {:5.2f} | ppl {:8.2f}'.format(
                                epoch, i, train_num_of_batches, scheduler.get_lr()[0],
                                elapsed * 1000 / log_interval,
                                cur_loss, math.exp(cur_loss)))
                    total_loss = 0
                    start_time = time.time()
                    if i % save_interval == 0:
                        with open('../log/check_point/' + self.model_name + '_model_log.txt', 'a') as ff:
                            ff.write('| epoch {:3d} | {:5d}/{:5d} batches | '
                            'lr {:02.2f} | ms/batch {:5.2f} | '
                            'loss {:5.2f} | ppl {:8.2f}\n'.format(
                             epoch, i, train_num_of_batches, scheduler.get_lr()[0],
                             elapsed * 1000 / log_interval,
                             cur_loss, math.exp(cur_loss)))

            val_loss = self.evaluate(eval_dataset)
            self.model.train()
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, math.exp(val_loss)))
            print('-' * 89)
            with open('../log/check_point/' + self.model_name + '_model_log.txt', 'a') as ff:
                ff.write('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}\n'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, math.exp(val_loss)))

            torch.save(model.state_dict(), '../log/pth/' + self.model_name + '_model_save.pth')
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), '../log/pth/' + self.model_name + '_best_model_save.pth')
                best_val_loss = val_loss
                best_model = model
            start_step = 0
            scheduler.step()

    def evaluate(self, dataset):
        self.model.eval()
        total_loss = 0.
        self.model.to(self.device)

        num_of_batches = int(len(dataset)/self.batch_size)
        with torch.no_grad():
            for i in range(num_of_batches):
                input_list=[]
                target_list=[]
                if self.use_sync_pos:
                    pos_list=[]
                    sync_pos_list=[]
                else:
                    pos_list=None
                    sync_pos_list=None
                data = dataset[i*batch_size:i*batch_size+batch_size]

                targets_mask = torch.Tensor()
                for indx, row in data.iterrows():
                    _input = row['token']
                    _target = row['target']

                    input_list.append(_input)
                    target_list.append(_target)
                    if self.use_sync_pos:
                        pos_list.append(list(map(int,row['pos'].split())))
                        sync_pos_list.append(list(map(int,row['sync_pos'].split())))
                inputs = tokenizer.batch_encode_plus(input_list, return_tensors='pt',
                        padding='max_length', truncation=True, max_length=400)
                targets = tokenizer.batch_encode_plus(target_list, return_tensors='pt',
                        padding='max_length', truncation=True, max_length=400)

                for tg in targets["input_ids"]:
                    targets_mask = torch.cat((targets_mask,
                        self.make_std_mask(tg, self.pad_token_idx).unsqueeze(0)),dim=0)
                
                inputs_mask = inputs["attention_mask"].unsqueeze(1)

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                generator_logit, loss = self.model.forward(inputs["input_ids"], targets["input_ids"],
                        inputs_mask.to(self.device), targets_mask.to(self.device), labels=targets["input_ids"],
                        pos=pos_list, sync_pos=sync_pos_list)

                total_loss += loss.item()

        return total_loss / (len(dataset) - 1)

if __name__ == '__main__':
    #data_name = 'translation'
    data_name = 'DeepFix'
    #vocab_path = '../vocab/vocab.model'
    train_path = '../data/' + data_name + '/train.csv'
    val_path = '../data/' + data_name + '/val.csv'
    vocab_path = '../vocab/vocab.txt'
    #train_path = '../data/' + data_name + '/test.csv'
    #val_path = '../data/' + data_name + '/test.csv'

    if not os.path.exists("../log/"):
        os.mkdir("../log/")
    if not os.path.exists("../log/pth/"):
        os.mkdir("../log/pth/")
    if not os.path.exists("../log/check_point"):
        os.mkdir("../log/check_point")

    # model setting
    sync_pos = args.sync_pos
    print("Start training whit synchronized position" if args.sync_pos else "Start training")
    model_name = 'transformer_' + data_name + ('_sync_pos' if sync_pos else '')
    #vocab_num = 1269
    vocab_num = 1270
    max_length = 400
    d_model = 200
    head_num = 8
    dropout = 0.1
    N = 6
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = BertTokenizer(vocab_path, do_lower_case=False, do_basic_tokenize=False)

    # hyper parameter
    epochs = 50
    batch_size = 64
    padding_idx = tokenizer.pad_token_id
    learning_rate = 0.01

    model = Transformer(vocab_num=vocab_num,
                          d_model=d_model,
                          max_seq_len=max_length,
                          head_num=head_num,
                          dropout=dropout,
                          N=N,
                          sync_pos=sync_pos)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    trainer = Trainer(tokenizer, model, max_length, device, model_name, batch_size, sync_pos)
    train, val = trainer.set_data(train_path, val_path)
    trainer.train(epochs, train, val, optimizer, scheduler)
