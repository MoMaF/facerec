#! /usr/bin/env python3

import jsonlines
import json

#m = 10035
#trajl = '10035-data/trajectories.jsonl'
#mm = str(m)

m = '332020'
trajl = 'out/'+m+'-data/trajectories.jsonl'
mm = str('{:03d}'.format(int(m[:-4])))+m[-4:]

clust = trajl.replace('/trajectories.jsonl', '/clusters.json')
clust = json.load(open(clust))['clusters']
#print(clust)

traj_i = 0
with jsonlines.open(trajl) as tr:
    for l in tr:
        name = mm+'_'+str(clust[traj_i])
        traj_i += 1
        # print(l)
        li = l['index']
        if True:
            s = l['start']
            b = l['bbs']
            for f in b:
                show = s>=82500 and s<84000
                show = True
                if show:
                    #z = re.match('momaf:elonet_henkilo_(\d+)', tra[li])
                    #assert z
                    #id = int(z.group(1))
                    #print(m, s, f, tra[li], act[id])
                    if True:
                        print('**boxdata** {} {} {} retinaface facenet {} {} {} {} 1 face {}'\
                              .format(mm, s, s+1, f[0], f[1], f[2], f[3], name))
                    s += 1
