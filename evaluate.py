import csv
import json

from attrdict import AttrDict
import torch.nn as nn
import torch
from transformers import (
  ElectraConfig,
  ElectraTokenizer,
)

from koelectra import koElectraForSequenceClassification, koelectra_input

labels = ["E10","E11","E12","E13","E14","E15","E16","E17","E18","E19","E20","E21","E22","E23","E24","E25","E26","E27","E28","E29","E30","E31","E32","E33","E34","E35","E36","E37","E38","E39","E40","E41","E42","E43","E44","E45","E46","E47","E48","E49","E50","E51","E52","E53","E54","E55","E56","E57","E58","E59","E60","E61","E62","E63","E64","E65","E66","E67","E68","E69"]

if __name__ == '__main__':
    config_path = './config/koelectra-small-v2.json'
    ckpt_path = './ckpt/koelectra-small-v2-mz-ckpt/checkpoint-11500/pytorch_model.bin'

    with open(config_path) as f:
        args = AttrDict(json.load(f))

    device = torch.device('cuda')

    checkpoint = torch.load(ckpt_path, map_location=device)

    tokenizer = ElectraTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    config = ElectraConfig.from_pretrained(args.model_name_or_path)
    model = koElectraForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            config=config,
            num_labels=60
            )

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    f = open('./final_test.csv', 'r', newline='') # f = open('final_test.csv', 'a', newline='')
    lines = csv.reader(f)
    next(lines)
    talks = []
    result = {}
    preds = []

    for line in lines:
        # talk_id = line[0]
        talk_id = line[1]
        talks.append(talk_id)

        # text = line[1]
        text = line[0]
        data = koelectra_input(tokenizer, text, device, 512)
        output = model(**data)

        logit = output
        softmax_logit = nn.Softmax(logit).dim
        softmax_logit = softmax_logit[0].squeeze()

        max_index = torch.argmax(softmax_logit).item()
        max_index_value = softmax_logit[torch.argmax(softmax_logit)].item()
        preds.append(max_index)
    f.close()

    for i in range(len(talks)):
        # result[talks[i]] = 'E' + str(preds[i])
        result[talks[i]] = str(labels[preds[i]])

    with open('result.json', 'w') as outputfile:
        json.dump(result, outputfile, indent=4)