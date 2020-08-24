import csv
import argparse
import os
import seaborn as sns
sns.set()
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--dataset',type=str, default='MultiWoZ')
args = parser.parse_args()

filepath = os.path.join(os.path.expanduser('~'),'Documents','Probing-Tests','Results',args.dataset,'Analsysis')
fp = open(filepath+'''/Probe_Tasks_'''+args.dataset+'''.csv''')
reader = csv.reader(fp)

def get_Data():
    data = {}
    target = open(os.path.join(filepath,"All_epoch_Probe_Tasks_"+args.dataset+".csv"), "a")
    models = ['bilstm_attn','seq2seq_attn','seq2seq','hred','transformer']
    seeds = ['100','101','102']
    epochs = [str(i) for i in range(25)]
    header = []
    i = 0
    for model in models:
        for seed in seeds:
            for epoch in epochs:
                if model not in data:
                    data.update({model:{}})
                if seed not in data[model]:
                    data[model].update({seed:{}})
                if str(epoch) not in data[model][seed]:
                    data[model][seed].update({str(epoch):{}})
    for row in reader:
        if i == 0:
            header = row
            i+=1
        else:
            for cols in header[3:]:
                if 'best' not in row[1]:
                    data[row[0]][row[2]][row[1]].update({cols:row[header.index(cols)]})
    #impute average
    fields = ['Model','Seed','Epoch']+header[3:]
    writer = csv.DictWriter(target, fieldnames=fields)
    writer.writerow(dict(zip(['Model','Seed','Epoch']+header[3:], ['Model','Seed','Epoch']+header[3:])))
    for model in models:
        for seed in ['100','101','102']:
            for epoch in range(0,25):
                dict_ = {}
                dict_['Model'] = model
                dict_['Seed'] = str(seed)
                dict_['Epoch'] = str(epoch)
                for cols in header[3:]:
                    dict_[cols] = float(data[model][seed][str(epoch)][cols])
                writer.writerow(dict_)
    return header

def createGraph(graphparam):
    sns.lineplot(x = 'Epoch', y =graphparam, style = 'Model',hue = 'Model',data = pd.read_csv(os.path.join(filepath,"All_epoch_Probe_Tasks_"+args.dataset+".csv")))
    plt.savefig(os.path.join(filepath,'plot_'+graphparam+'.png'))
    plt.close()


if __name__ == '__main__':
    header = get_Data()
    for cols in header[3:]:
        createGraph(cols)
