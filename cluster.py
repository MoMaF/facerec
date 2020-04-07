#! /usr/bin/env python3

import scipy.cluster as cluster
import matplotlib.pyplot as plt

#method='single'
method='complete'
#method=''
#method=''

val=20
criterion='maxclust'

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

n = len(dat)
#n = 100

s = [ i for i in range(n) ]
sel = []
for i in range(len(s)):
    sel.append(dat[s[i]])

link = cluster.hierarchy.linkage(sel, method=method)
print(link)
#plt.figure()
#dn = cluster.hierarchy.dendrogram(link)
#plt.show()

fc = cluster.hierarchy.fcluster(link, val, criterion=criterion)
print(fc)

for i in range(val):
    print(r'LATEX \clusterbegins{',i+1,'}', sep='') 
    #print(i+1, end=' ')
    for j, k in enumerate(fc):
        if k==i+1:
            print(r'LATEX \clusterentry{',lab[j],'}', sep='') 
            #print(lab[j], end=' ')
    #print()
    print(r'LATEX \clusterends{',i+1,'}', sep='') 

    
