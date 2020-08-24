import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default='PersonaChat')
args = parser.parse_args()

ave = csv.reader(open('../Results/'+args.dataset+'/Analsysis/'+args.dataset+'_ave.csv','r'))
var = csv.reader(open('../Results/'+args.dataset+'/Analsysis/'+args.dataset+'_var.csv','r'))

fp = open(args.dataset+'_table.txt','w')
for a,v in zip(ave,var):
    s = ''
    assert len(a) == len(v)
    for i in range(len(a)):
        if i == 0:
            s += a[i] + '&'
        else:
            s+='{:.2f} $\pm$ {:3.2f} &'.format(float(a[i])*100, float(v[i]) * 100)
    fp.write(s+'\n\n')
