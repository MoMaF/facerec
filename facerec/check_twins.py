#! /usr/bin/env python3

import math
import json
import pandas
import argparse
import re
import glob

#dir = '/scratch/project_2002528'
dir = '/scratch/project_462000139/jorma/momaf'

parser = argparse.ArgumentParser(allow_abbrev=True,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Generaate ASS subtitles of face recognitions')
parser.add_argument("--path", type=str,
                    help="Path to data directory for a film.")
parser.add_argument('--debug', action='store_true',
                    help='show debug output instead of RDF')
parser.add_argument('--datadir', type=str, default=dir+"/emil/data",
                    help="Directory where <movieid>-data directories reside, default=%(default)s")
args = parser.parse_args()

assert args.path is not None, '--path was undefined'

m = re.search('/(\d+)-data', args.path)
assert m, '--path should specify directory like /12345-data'
m = int(m.group(1))

sw  = None
sh  = None
dw  = None
dh  = None
sar = None
fps = None

jf = dir+'/metadata/'+str(m)+'-*.json'
j  = glob.glob(jf)
# assert len(j)==1, 'not unique file <'+jf+'> '+str(j)
assert len(j)>0, 'metadata file <'+jf+'> not found'
meta = json.load(open(j[0]))
assert 'streams' in meta
for s in meta['streams']:
    if s['codec_type']=='video':
        f = s['avg_frame_rate']
        x = re.match('(\d+)/(\d+)', f)
        if x:
            assert int(x.group(1))>0 and int(x.group(2))>0
            fps = int(x.group(1))/int(x.group(2))
        else:
            fps = float(f)
assert fps is not None

if args.debug:
    print('sw={} sh={} dw={} dh={} sar={} f={} x={} fps={}'.
          format(sw, sh, dw, dh, sar, f, x, fps))

def tx(f):
    s = f/fps
    h = int(math.floor(s/3600))
    s -= 3600*h
    m = int(math.floor(s/60))
    s -= 60*m
    return '{}:{}:{:.2f}'.format(h, m, s)

cl = json.load(open(args.path+'/clusters.json'))
cl = cl['clusters']

pr = json.load(open(args.path+'/predictions.json'))
pr = pr['predictions']

ac = pandas.read_csv('actors.csv')

i = 0

faces = {}

with open(args.path+'/trajectories.jsonl') as f:
    for l in f:
        l = json.loads(l)
        #print(l)
        s = l['start']
        t = cl[i]
        t = pr[str(t)]
        t = [ (v, k) for k, v in t.items() ]
        t.sort()
        #print('a', t)
        t = t[-1][1]
        t = t.split('_')
        t = int(t[-1])
        #t = list(t)
        t = ac[ac['id']==t].iloc[0]['name']
        #print('b', t)
        
        for b in l['bbs']:
            # print(tx(s), s, t)
            if s not in faces:
                faces[s] = {}
            if t not in faces[s]:
                faces[s][t] = 0
            faces[s][t] += 1
            s += 1
        i += 1
        
        
for i, f in faces.items():
    for a, n in f.items():
        if n!=1:
            print(tx(i), i, a, n)

        
