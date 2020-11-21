import json
import csv

labels = ["E10","E11","E12","E13","E14","E15","E16","E17","E18","E19","E20","E21","E22","E23","E24","E25","E26","E27","E28","E29","E30","E31","E32","E33","E34","E35","E36","E37","E38","E39","E40","E41","E42","E43","E44","E45","E46","E47","E48","E49","E50","E51","E52","E53","E54","E55","E56","E57","E58","E59","E60","E61","E62","E63","E64","E65","E66","E67","E68","E69"]

# train.csv 생성
def create_train():
    with open('train_data.json', 'r', encoding="utf-8") as f:
        json_data = json.load(f)

    f = open('train.csv', 'a', newline='')
    title = ['label', 'content']
    wr = csv.writer(f)
    wr.writerow(title)

    for i in json_data:
        tmp = i['talk']['content']
        label = i['profile']['emotion']['type']
        tmp2 = []
        tmp2.append(int(labels.index(label)))

        string = tmp['HS01'] + ' [SEP] ' + tmp['SS01'] + ' [SEP] ' + tmp['HS02']
        tmp2.append(string)
        wr = csv.writer(f)
        wr.writerow(tmp2)

# dev.csv 생성
def create_dev():
    with open('dev_data.json', 'r', encoding="utf-8") as f:
        json_data = json.load(f)

    f = open('dev.csv', 'a', newline='')
    title = ['label', 'content']
    wr = csv.writer(f)
    wr.writerow(title)

    for i in json_data:
        tmp = i['talk']['content']
        label = i['profile']['emotion']['type']
        tmp2 = []
        tmp2.append(int(labels.index(label)))

        string = tmp['HS01'] + ' [SEP] ' + tmp['SS01'] + ' [SEP] ' + tmp['HS02']
        tmp2.append(string)

        wr = csv.writer(f)
        wr.writerow(tmp2)

# test.csv 생성
def create_test():
    # with open('test_data.json', 'r', encoding="utf-8") as f:
    with open('final_test.json', 'r', encoding="utf-8") as f:
        json_data = json.load(f)

    f = open('final_test.csv', 'a', newline='')
    title = ['content', 'talkid']
    wr = csv.writer(f)
    wr.writerow(title)

    for i in json_data:
        tmp = i['talk']['content']
        id = i['talk']['id']['talk-id']
        tmp2 = []

        string = tmp['HS01'] + ' [SEP] ' + tmp['SS01'] + ' [SEP] ' + tmp['HS02']
        tmp2.append(string)
        tmp2.append(id)

        wr = csv.writer(f)
        wr.writerow(tmp2)

if __name__ == "__main__":
    # create_train() # train.csv 생성
    # create_dev() # dev.csv 생성
    create_test() # test.csv 생성
