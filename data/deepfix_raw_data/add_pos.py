import csv
from tqdm import tqdm
import argparse

par = argparse.ArgumentParser()
par.add_argument("-d", "--data_path", default='test',
                 type=str, help="Please enter data path")
args = par.parse_args()

with open(args.data_path + '.csv','r') as f:
    rdr = csv.reader(f)

    insert_tok = list(range(1,422))
    max_length = 400

    with open(args.data_path + '_2.csv', 'w', newline='') as new_f:
        wr = csv.writer(new_f)
        for i, line in tqdm(enumerate(rdr)):
            if i == 0:
                line.append('pos')
                line.append('sync_pos')
                wr.writerow(line)
                continue
            pos = list()
            sync_pos = list()
            target = line[2].split()
            length = len(line[1].split())
            current_length = len(line[1].split())
            for m in range(max_length):
                if length - m > 0:
                    pos.append(str(length - m))
                else:
                    pos.append(str(0))
                if current_length <= 0:
                    sync_pos.append(str(0))
                else:
                    if int(target[m]) not in insert_tok:
                        current_length -= 1
                    sync_pos.append(str(current_length+1))

            line.append(' '.join(pos))
            line.append(' '.join(sync_pos))
            wr.writerow(line)
