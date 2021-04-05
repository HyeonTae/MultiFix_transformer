import torch
from transformer import Transformer
from transformers import BertTokenizer
from dataset import TranslationDataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import math
import os
import time


class TranslationTrainer():
    def __init__(self,
                 dataset,
                 tokenizer,
                 model,
                 max_len,
                 device,
                 model_name,
                 checkpoint_path,
                 batch_size,
                 ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model = model
        self.max_len = max_len
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.ntoken = tokenizer.vocab_size
        self.batch_size = batch_size

    def my_collate_fn(self, samples):
        input_str =[]
        target_str =[]
        input_str.append([sample['input_str'] for sample in samples])
        input = [sample['input'] for sample in samples]
        input_mask = [sample['input_mask'] for sample in samples]
        target = [sample['target'] for sample in samples]
        target_mask = [sample['target_mask'] for sample in samples]
        token_num = [sample['token_num'] for sample in samples]
        target_str.append([sample['target_str'] for sample in samples])

        return {
            "input_str":input_str,
            "input":torch.stack(input).contiguous(),                                              # input
            "input_mask": torch.stack(input_mask).contiguous(),       # input_mask
            "target": torch.stack(target).contiguous(),                                           # target,
            "target_mask": torch.stack(target_mask).contiguous(),   # target_mask
            "token_num": torch.stack(token_num).contiguous(),   # token_num
            "target_str": target_str
        }

    def build_dataloaders(self, train_test_split=0.1, train_shuffle=True, eval_shuffle=True):
        dataset_len = len(self.dataset)
        eval_len = int(dataset_len * train_test_split)
        train_len = dataset_len - eval_len
        train_dataset, eval_dataset = random_split(self.dataset, (train_len, eval_len))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=train_shuffle, collate_fn=self.my_collate_fn)
        eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=eval_shuffle , collate_fn=self.my_collate_fn)

        return train_loader, eval_loader

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

            pb = tqdm(enumerate(train_dataset),
                      desc=f'Epoch-{epoch} Iterator',
                      total=train_dataset_length)
                      #bar_format='{l_bar}{bar:10}{r_bar}'
                      #)
            pb.update(start_step)

            for i,data in pb:
                if i < start_step:
                    continue

                input = data['input'].to(self.device)
                target = data['target'].to(self.device)
                input_mask = data['input_mask'].to(self.device)
                target_mask = data['target_mask'].to(self.device)

                optimizer.zero_grad()
                generator_logit, loss = self.model.forward(input, target, input_mask, target_mask, labels=target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                losses[global_steps] = loss.item()
                total_loss += loss.item()
                log_interval = 1
                save_interval = 500

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
                                epoch, i, len(train_dataset), scheduler.get_lr()[0],
                                elapsed * 1000 / log_interval,
                                cur_loss, math.exp(cur_loss)))
                    total_loss = 0
                    start_time = time.time()
                    # self.save(epoch, self.model, optimizer, losses, global_steps)
                    if i % save_interval == 0:
                        self.save(epoch, self.model, optimizer, losses, global_steps)

            val_loss = self.evaluate(eval_dataset)
            self.model.train()
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, math.exp(val_loss)))
            print('-' * 89)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
            start_step = 0
            scheduler.step()

    def evaluate(self, dataset):
        self.model.eval()
        total_loss = 0.

        self.model.to(self.device)
        with torch.no_grad():
            for i, data in enumerate(dataset):
                input = data['input'].to(self.device)
                target = data['target'].to(self.device)
                input_mask = data['input_mask'].to(self.device)
                target_mask = data['target_mask'].to(self.device)

                generator_logit, loss = self.model.forward(input, target, input_mask, target_mask, labels=target)
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
    data_name = 'DrRepair_deepfix'
    vocab_path = '../data_processing/vocab.txt'
    data_path = '../data/' + data_name + '/test_2.txt'
    checkpoint_path = '../log/check_point'

    # model setting
    model_name = 'transformer_' + data_name + '_sin'
    vocab_num = 1076
    max_length = 500
    d_model = 512
    head_num = 8
    dropout = 0.1
    N = 6
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=False)

    # hyper parameter
    epochs = 100
    batch_size = 4
    padding_idx = tokenizer.pad_token_id
    learning_rate = 0.01

    dataset = TranslationDataset(tokenizer=tokenizer, file_path=data_path, max_length=max_length)

    model = Transformer(vocab_num=vocab_num,
                          d_model=d_model,
                          max_seq_len=max_length,
                          head_num=head_num,
                          dropout=dropout,
                          N=N)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    trainer = TranslationTrainer(dataset, tokenizer, model, max_length, device, model_name, checkpoint_path, batch_size)

    train_dataloader, eval_dataloader = trainer.build_dataloaders(train_test_split=0.2)

    trainer.train(epochs, train_dataloader, eval_dataloader, optimizer, scheduler)