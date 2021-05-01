import torch
from transformer import Transformer
from transformers import T5Tokenizer
from tqdm import tqdm
import pandas as pd
import math
import os
import time


class Trainer():
    def __init__(self,
                 tokenizer,
                 model,
                 max_len,
                 device,
                 model_name,
                 checkpoint_path,
                 batch_size,
                 ):
        self.tokenizer = tokenizer
        self.model = model
        self.max_len = max_len
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.batch_size = batch_size

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
        if os.path.isfile(f'{self.checkpoint_path}/{self.model_name}.pth'):
            checkpoint = torch.load(f'{self.checkpoint_path}/{self.model_name}.pth', map_location=self.device)
            start_epoch = checkpoint['epoch']
            losses = checkpoint['losses']
            global_steps = checkpoint['train_step']
            start_step = global_steps if start_epoch == 0 else (global_steps % train_dataset_length) + 1

            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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
                data = train_dataset[i*batch_size:i*batch_size+batch_size]
                for indx, row in data.iterrows():
                    _input = row['token']
                    _target = row['target']
                    input_list.append(_input)
                    target_list.append(_target)
                inputs = tokenizer.batch_encode_plus(input_list, return_tensors='pt',
                        padding='max_length', truncation=True, max_length=400)
                targets = tokenizer.batch_encode_plus(target_list, return_tensors='pt',
                        padding='max_length', truncation=True, max_length=400)

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                generator_logit, loss = self.model.forward(inputs["input_ids"], targets["input_ids"],
                        inputs["attention_mask"], targets["attention_mask"], labels=targets["input_ids"])

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
                    # print('| epoch {:3d} | {:5d}/{:5d} batches | '
                    #       'lr {:02.2f} | ms/batch {:5.2f} | '
                    #       'loss {:5.2f} | ppl {:8.2f}'.format(
                    #         epoch, i, len(train_dataset), scheduler.get_lr()[0],
                    #         elapsed * 1000 / log_interval,
                    #         cur_loss, math.exp(cur_loss)))
                    pb.set_postfix_str('| epoch {:3d} | {:5d}/{:5d} batches | '
                                     'lr {:02.2f} | ms/batch {:5.2f} | '
                                     'loss {:5.2f} | ppl {:8.2f}'.format(
                                epoch, i, train_num_of_batches, scheduler.get_lr()[0],
                                elapsed * 1000 / log_interval,
                                cur_loss, math.exp(cur_loss)))
                    total_loss = 0
                    start_time = time.time()
                    # self.save(epoch, self.model, optimizer, losses, global_steps)
                    if i % save_interval == 0:
                        with open('../log/check_point/sync_model_log.txt', 'a') as ff:
                            ff.write('| epoch {:3d} | {:5d}/{:5d} batches | '
                            'lr {:02.2f} | ms/batch {:5.2f} | '
                            'loss {:5.2f} | ppl {:8.2f}\n'.format(
                             epoch, i, train_num_of_batches, scheduler.get_lr()[0],
                             elapsed * 1000 / log_interval,
                             cur_loss, math.exp(cur_loss)))
                        #self.save(epoch, self.model, optimizer, losses, global_steps)

            val_loss = self.evaluate(eval_dataset)
            self.model.train()
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, math.exp(val_loss)))
            print('-' * 89)
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), '../log/pth/sync_model_save.pth')
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
                data = dataset[i*batch_size:i*batch_size+batch_size]
                for indx, row in data.iterrows():
                    _input = row['token']
                    _target = row['target']
                    input_list.append(_input)
                    target_list.append(_target)
                inputs = tokenizer.batch_encode_plus(input_list, return_tensors='pt',
                        padding='max_length', truncation=True, max_length=400)
                targets = tokenizer.batch_encode_plus(target_list, return_tensors='pt',
                        padding='max_length', truncation=True, max_length=400)

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                generator_logit, loss = self.model.forward(inputs["input_ids"], targets["input_ids"],
                        inputs["attention_mask"], targets["attention_mask"], labels=targets["input_ids"])
                total_loss += loss.item()

        return total_loss / (len(dataset) - 1)

    def save(self, epoch, model, optimizer, losses, train_step):
        torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'losses': losses,
              'train_step': train_step,
               }, f'{self.checkpoint_path}/{self.model_name}.pth')


if __name__ == '__main__':
    data_name = 'DeepFix'
    vocab_path = '../vocab/vocab.model'
    train_path = '../data/' + data_name + '/train.csv'
    val_path = '../data/' + data_name + '/val.csv'
    #train_path = '../data/' + data_name + '/test.csv'
    #val_path = '../data/' + data_name + '/test.csv'
    checkpoint_path = '../log/check_point'

    # model setting
    model_name = 'transformer_' + data_name + '_sin'
    vocab_num = 1267
    max_length = 400
    #d_model = 300
    d_model = 200
    head_num = 8
    #head_num = 4
    dropout = 0.1
    N = 6
    #N = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = T5Tokenizer(vocab_path)

    # hyper parameter
    epochs = 100
    batch_size = 64
    padding_idx = tokenizer.pad_token_id
    learning_rate = 0.01

    model = Transformer(vocab_num=vocab_num,
                          d_model=d_model,
                          max_seq_len=max_length,
                          head_num=head_num,
                          dropout=dropout,
                          N=N,
                          sync_pos=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    trainer = Trainer(tokenizer, model, max_length, device, model_name, checkpoint_path, batch_size)
    train, val = trainer.set_data(train_path, val_path)
    trainer.train(epochs, train, val, optimizer, scheduler)
