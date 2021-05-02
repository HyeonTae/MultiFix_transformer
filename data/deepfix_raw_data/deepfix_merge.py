import csv
from tqdm import tqdm

path = ['data/DeepFix/ids/', 'data/DeepFix/typo/']

def csv_merge(path1, path2, tp):
    with open('data/DeepFix/' + tp + '.csv', 'w') as csv_out_file:
        fwriter = csv.writer(csv_out_file)
        with open(path1 + tp + '.csv', 'r') as file1:
            freader = csv.reader(file1)
            for row in tqdm(freader):
                fwriter.writerow(row)
        with open(path2 + tp + '.csv', 'r') as file2:
            freader = csv.reader(file2)
            header = next(freader)
            for row in tqdm(freader):
                fwriter.writerow(row)

print("Train csv file merge..")
csv_merge(path[0], path[1], 'train')
print("Validation csv file merge..")
csv_merge(path[0], path[1], 'val')
print("done..")
