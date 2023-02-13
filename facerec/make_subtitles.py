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
        assert sw is None and sh is None and fps is None
        sw = int(s['width'])
        sh = int(s['height'])
        a = s.get('sample_aspect_ratio', None)
        if a is None:
            sar = 1
        else:
            x = re.match('(\d+):(\d+)', a)
            if x:
                assert int(x.group(1))>0 and int(x.group(2))>0
                sar = int(x.group(1))/int(x.group(2))
            else:
                sar = float(a)
        assert sar>=1
        dw = int(sar*sw)
        dh = sh
        f = s['avg_frame_rate']
        x = re.match('(\d+)/(\d+)', f)
        if x:
            assert int(x.group(1))>0 and int(x.group(2))>0
            fps = int(x.group(1))/int(x.group(2))
        else:
            fps = float(f)
assert sw is not None and sh is not None and fps is not None
assert sw>0 and sh>0 and dw>0 and dh>0 and fps and fps>0

fname = meta['format']['filename']
fname = fname.split('/')[-1]
fname = ''.join(fname.split('.')[:-1])+'.ass'
print(f'Writing subtitles in {fname}')

if args.debug:
    print('sw={} sh={} dw={} dh={} sar={} f={} x={} fps={}'.
          format(sw, sh, dw, dh, sar, f, x, fps))

header = \
"""[Script Info]
Title:
Original Script:
Original Translation:
Original Editing:
Original Timing:
Original Script Checking:
ScriptType: v4.00+
Collisions: Normal
PlayResX: PLAYRESX
PlayResY: PLAYRESY
PlayDepth: 0
Timer: 100,0000
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: objectbox,DejaVu Sans,80,&HFF000000,&H00B4FCFC,&H00000000,&H00000000,0,0,0,0,100,100,0.00,0.00,1,2.50,0.00,7,50,50,50,0
Style: objecttxt,DejaVu Sans,40,&H000000FF,&H00B4FCFC,&H00FF0000,&H00FF0000,0,0,0,0,100,100,0.00,0.00,1,2.50,0.00,7,50,50,50,0
Style: top-left,DejaVu Sans,25,&H00FFFFFF,&H00B4FCFC,&H00000000,&H00000000,0,0,0,0,100,100,0.00,0.00,1,2.50,0.00,7,50,50,50,0

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"""

header = header.replace('PLAYRESX', str(dw))
header = header.replace('PLAYRESY', str(dh))

def tx(f):
    s = f/fps
    h = int(math.floor(s/3600))
    s -= 3600*h
    m = int(math.floor(s/60))
    s -= 60*m
    return '{}:{}:{:.2f}'.format(h, m, s)

def boxtext(f, b, t, fp):
    s = tx(f)
    e = tx(f+1)
    m = 1/sar
    print(r'Dialogue: 1,{},{},objectbox,foo,000,000,000,,{{\pos(0,0)\p1\3c&H0000FF&}}m {} {} l {} {} {} {} {} {}{{\p0\r}}'.
          format(s, e, m*b[0], m*b[1], m*b[2], m*b[1], m*b[2], m*b[3], m*b[0], m*b[3]), file=fp)
    print(r'Dialogue: 1,{},{},objecttxt,foo,000,000,000,,{{\pos({},{})\an5\1c&HFFFFFF&}}{}{{\r}}'.
          format(s, e, (b[0]+b[2])/2, b[1], t), file=fp)

cl = json.load(open(args.path+'/clusters.json'))
cl = cl['clusters']

pr = json.load(open(args.path+'/predictions.json'))
pr = pr['predictions']

ac = pandas.read_csv('actors.csv')

tr = []
i = 0

fp = open(fname, 'w')
print(header, file=fp)

with open(args.path+'/trajectories.jsonl') as f:
    for l in f:
        l = json.loads(l)
        s = l['start']
        t = cl[i]
        clu = str(t)
        t = pr[clu]
        t = [ (v, k) for k, v in t.items() ]
        #print('a', t)
        t.sort()
        t = t[-1][1]
        t = t.split('_')
        t = int(t[-1])
        #t = list(t)
        t = ac[ac['id']==t].iloc[0]['name']
        #print('b', t)
        
        for b in l['bbs']:
            boxtext(s, b, t+"/"+clu, fp)
            s += 1
        i += 1
        
        
