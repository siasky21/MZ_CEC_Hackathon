import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import ElectraTokenizer

class MzTextClassificationDataset(Dataset):
    def __init__(self,
                 file_path = "train.csv",
                 num_label = 60,
                 device = 'cpu',
                 max_seq_len = 512,
                 tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-small-v2-discriminator')
                 ):
        self.file_path = file_path
        self.device = device
        self.tokenizer = tokenizer # if tokenizer is not None else get_tokenizer()
        self.data = []

        f = open(self.file_path, 'r')

        lines = csv.reader(f)
        next(lines)
        for line in lines:
            context = tokenizer.encode(line[1])
            token_type_ids = [0] * len(context)
            attention_mask = [1] * len(context)

            padding_length = max_seq_len - len(context)

            context += [0] * padding_length
            token_type_ids += [0] * padding_length
            attention_mask += [0] * padding_length

            label = int(line[0])

            data = {
                    'input_ids' : torch.tensor(context).to(self.device),
                    'token_type_ids': torch.tensor(token_type_ids).to(self.device),
                    'attention_mask': torch.tensor(attention_mask).to(self.device),
                    'labels': torch.tensor(label).to(self.device),
            }

            self.data.append(data)

        f.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return item

if __name__ == "__main__":
    dataset = MzTextClassificationDataset()
    for i in dataset:
        print(len(i['input_ids']))
        print(len(i['token_type_ids']))
        print(len(i['attention_mask']))
        print(i['labels'])
        break