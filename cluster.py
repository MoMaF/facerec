#! /usr/bin/env python3

import scipy.cluster as cluster
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

#method='single'
method='complete'
#method=''
#method=''

n         = 10000
val       =   100
max_m     =    35
criterion = 'maxclust'

dat = []
lab = []
with open('features.dat') as f:
    for l in f:
        ll = l.split(' ')
        lx = ll.pop()
        ll = [float(f) for f in ll]
        #print(ll)
        dat.append(ll)
        lab.append(lx.strip())

# s = [ i for i in range(n) ]
s = np.random.permutation(len(dat))[:n]
# print(s)
# exit(0)

sel = []
for i in range(len(s)):
    sel.append(dat[s[i]])

link = cluster.hierarchy.linkage(sel, method=method)
#print(link)
#plt.figure()
#dn = cluster.hierarchy.dendrogram(link)
#plt.show()

fc = cluster.hierarchy.fcluster(link, val, criterion=criterion)
#print(fc)

for i in range(val):
    x = []
    y = []
    for j, k in enumerate(fc):
        if k==i+1:
            x.append(sel[j])
            y.append(lab[s[j]])

    m = np.mean(x, axis=0)
    #print(m)
    #print(type(x-m), (x-m).shape)
    d = np.linalg.norm(x-m, axis=1)
    #print(d)
    #print(y)
    d = zip(range(len(d)), y, d)
    e = sorted(d, key=lambda a: a[2])
    #print(e)

    eff_m = len(e)//2
    if eff_m>max_m:
        eff_m = max_m
    
    print(r'LATEX \clusterbegins{',i+1,' total ', len(x), '}', sep='') 
    for j in range(eff_m):
        print(r'LATEX \clusterentry{',e[j][1],'}', sep='') 
        print(r'LABEL [', i+1, '] ', e[j][1], sep='') 

    print(r'LATEX \clusterends{',i+1,'}', sep='') 

    
