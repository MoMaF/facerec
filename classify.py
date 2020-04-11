#! /usr/bin/env python3

from sklearn import svm
import numpy as np
import re

c2a = {}

with  open('ts-clusters-1.tsv') as f:
    for l in f:
        m = re.match('(\d+)\s+(.+)', l)
        assert m, l
        c = m.group(1)
        a = m.group(2)
        if a=='?' or a=='x':
            continue
        #print(c, a)
        c2a[c] = a

i2a = sorted(list(set(c2a.values())))
a2i = {}
print(i2a)
a2i = { y: x for x, y in enumerate(i2a) }

f2i = {}

with  open('clusters_labels.txt') as f:
    for l in f:
        m = re.match('LABEL\s+\[(\d+)\]\s+(.+)', l)
        assert m, l
        c = m.group(1)
        x = m.group(2)
        if c in c2a:
            #print(c2a[c], a2i[c2a[c]], x)
            f2i[x] = a2i[c2a[c]]
Xt = []
yt = []
Xa = []
ya = []

with  open('features.dat') as f:
    for l in f:
        l = l.strip()
        v = l.split(' ')
        x = v.pop()
        v = [ float(z) for z in v ]
        #print(x)
        if x in f2i:
            #print(x, f2i[x])
            Xt.append(v)
            yt.append(f2i[x])
        Xa.append(v)
        ya.append(x)

clf = svm.LinearSVC(max_iter=10000)
clf.fit(Xt, yt)

dec = clf.decision_function(Xa)
print(dec.shape)
dec = dec/2+0.5
#print(dec)

for i, r in enumerate(dec):
    rx = r.copy()
    z1 = np.argmax(rx)
    s1 = rx[z1]
    rx[z1] = -10000
    z2 = np.argmax(rx)
    s2 = rx[z2]
    s  = s1-s2
    
    # if z!=yt[i]:
        # print(i, z, yt[i])
    #print(ya[i], i2a[z])
    m = re.match('.*(\d{6}):(\d+)_(\d+)_(\d+)_(\d+)_(\d+)', ya[i])
    assert m, ya[i]
    f = int(m.group(2))
    print('*boxdata*', m.group(1), f , f+1, 'mtcnn', 'facenet',
          m.group(3), m.group(4), m.group(5), m.group(6), s, i2a[z1])
