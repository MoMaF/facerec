#! /usr/bin/env python3

import os
import re
import stat
import argparse
import subprocess
import tempfile
from datetime import datetime
from subprocess import PIPE, STDOUT

momaf_dir       =  '/scratch/project_462000139/jorma/momaf'
python_base_dir = f'{momaf_dir}/github/facerec/python_base'

_macros_ = { 'GPU-P': 'small-g',
             'CPU-P': 'small'    }

_default_ = { 'account':       'project_462000139',
              'job-name':      'JOB-NAME',
              'output':        'logs/slurm-FILMID-JOB-NAME-%A_%a.out' }

_partition_ = { 'small-g': { 'gpus-per-node': '1',
                             'ntaska':        '1',
                             'cpus-per-task': '1',
                             'mem-per-cpu':   '8000',
                             'time':          '03:00:00'},
                'small':   { 'mem-per-cpu':   '8000',
                             'time':          '03:00:00'}}

_stages_ = [ { 'name':      'extract',
               'partition': 'GPU-P',
               'array':     '0-99',
               'setup':     'tensorflow',
               'command':   './facerec/extract.py --n-shards $SLURM_ARRAY_TASK_COUNT'+\
                            ' --shard-i $SLURM_ARRAY_TASK_ID --save-every 5 --noimages'+\
                            ' --out-path ARGS.OUT_PATH ARGS.FILMFILE',
               'cleanup':   'echo Good bye cruel world!',
               'skip': ''
             },

             { 'name':       'merge',
               'setup':      'tensorflow',
               'command':    './facerec/merge_shards.py --path ARGS.OUT_PATH/FILMID-data'
             },

             { 'name':       'cluster',
               'setup':      'tensorflow',
               'command':    './facerec/cluster.py --path ARGS.OUT_PATH/FILMID-data'
             },
             
             { '': '', '': '', '': '', '': '', '': '', '': '' },
             { '': '', '': '', '': '', '': '', '': '', '': '' },
             { '': '', '': '', '': '', '': '', '': '', '': '' },
             { '': '', '': '', '': '', '': '', '': '', '': '' },
             { '': '', '': '', '': '', '': '', '': '', '': '' },
             { '': '', '': '', '': '', '': '', '': '', '': '' },
             { '': '', '': '', '': '', '': '', '': '', '': '' },
             { '': '', '': '', '': '', '': '', '': '', '': '' },
             { '': '', '': '', '': '', '': '', '': '', '': '' },
             { '': '', '': '', '': '', '': '', '': '', '': '' }]

_setup_ = { 'tensorflow': [  'export OMP_NUM_THREADS=1',
                             'export MPICH_GPU_SUPPORT_ENABLED=1',
                             'module use /appl/local/csc/modulefiles',
                             'module load tensorflow',
                            f'export PYTHONUSERBASE={python_base_dir}' ] }

sbatch_keys = [ 'account', 'acctg-freq', 'array', 'batch', 'clusters',
                'constraint', 'container', 'core-spec', 'cpus-per-gpu',
                'verbose', 'delay-boot', 'distribution', 'exclusive', 'export',
                'get-user-env', 'gpu-bind', 'gpu-freq', 'gpus', 'gpus-per-node',
                'gpus-per-task', 'gres', 'gres-flags', 'hint', 'ignore-pbs',
                'job-name', 'mem-bind', 'mem-per-cpu', 'mem-per-gpu', 'mem',
                'network', 'no-kill', 'no-requeue', 'open-mode', 'overcommit',
                'partition', 'power', 'profile', 'qos', 'requeue', 'reservation',
                'signal', 'spread-job', 'thread-spec', 'threads-per-core', 'time',
                'use-min-nodes', 'wait', 'wait-all-nodes', 'wckey', 'output']

def expand(s):
    while True:
        hit = False
        for k, v in _macros_.items():
            i = s.find(k)
            if i>=0:
                hit = True
                s = s[:i]+v+s[i+len(k):]
        if not hit:
            break
    return s


def submit_sbatch_and_wait(s):
    print(f'{datetime.now()}   Submitting batch job {s}')
    return False, ''


def run_script(s):
    print(f'{datetime.now()}   Running script {s}')
    os.chmod(s, stat.S_IRUSR | stat.S_IXUSR)
    # r = subprocess.run(f'/bin/ls -l {s}', shell=True)
    # r = subprocess.run(f'/bin/cat {s}', shell=True)
    r = subprocess.run(s, shell=True, stdout=PIPE, stderr=STDOUT)
    print(f'{datetime.now()}   Script {s} finished with {r.returncode}')
    return r.returncode==0, r.stdout
    

def show_script(s):
    print('================ script file begins ==============')
    print(open(s).read())
    print('================ script file ends ================')


def show_output(s):
    print('================== output begins ================')
    print(open(s).read())
    print('================== output ends ==================')


def run_stage(stage):
    #print(stage)
    name = stage['name']

    if 'skip' in stage:
        print(f'{datetime.now()}   Skipping stage <{name}>')
        return True
    
    _macros_['JOB-NAME'] = 'face-'+name
    s = {}
            
    partition = expand(stage['partition']) if 'partition' in stage else None

    if partition:
        if partition not in _partition_:
            print(f'No specs for partition <{partition}> in sbatch stage <{name}>')
            return False
    
        for k, v in _default_.items():
            s[k] = expand(v)
        for k, v in _partition_[partition].items():
            s[k] = expand(v)

    for k, v in stage.items():
        if k not in ['name', 'method']:
            s[k] = expand(v)

    #print(s)
    
    script = [ '#! /bin/bash', '' ]

    if partition is not None:
        for i in sbatch_keys:
            if i in s:
                script.append(f'#SBATCH --{i}={s[i]}')
        script.append('')

    if 'setup' in s:
        if s['setup'] not in _setup_:
            print(f'No specs for setup <{s["setup"]}> in sbatch stage <{name}>')
            return False
        for i in _setup_[s['setup']]:
            script.append(expand(i))
        script.append('')

    if 'command' not in s:
        print(f'No command in stage <{name}>')
        return False
        
    script.append(s['command'])
    script.append('')

    if 'cleanup' in s:
        script.append(expand(s['cleanup']))
        script.append('')
        
    sfile = tempfile.NamedTemporaryFile(mode='w+')
        
    print('\n'.join(script), file=sfile, flush=True)
    sfile.file.close()
    
    if partition is not None:
        r, t = submit_sbatch_and_wait(sfile.name)    
        if not r:
            print(f'{datetime.now()} Running sbatch job failed in stage <{name}>')
            show_script(sfile.name)
            return False
    else:
        r, t = run_script(sfile.name)    
        if not r:
            print(f'{datetime.now()} Running script failed in stage <{name}>')
            show_script(sfile.name)
            show_output(t)
            return False

    return True


def main(args):
    if not os.path.isfile(args.filmfile):
        print(f'<{args.filmfile}> is not a file')
        return False

    film = args.filmfile.split('/')[-1]
    
    m = re.search('(\d+)', film)
    if not m:
        print(f'No numbers in film name <{args.filmfile}>')
        return False
    
    film = m.group(1)

    _macros_['ARGS.FILMFILE'] = args.filmfile
    _macros_['ARGS.OUT_PATH'] = args.out_path 
    _macros_['FILMID']        = film

    if len(_stages_)==0:
        print(f'No stages defined, aborting.')
        return False
    
    for si, stage in enumerate(_stages_):
        if 'name' not in stage:
            ok = False
            print(f'{datetime.now()} Stage #{si} has no name, aborting.')
            return False

        start_time = datetime.now()
        print(f'{start_time} Starting stage #{si} <{stage["name"]}> for film <{film}>')
        
        ok = run_stage(stage)
        end_time = datetime.now()
        diff_time = end_time-start_time
        if not ok:
            print(f'{end_time} Stage #{si} <{stage["name"]}> failed in {diff_time} s, aborting.')
            return False

        print(f'{datetime.now()} Stage #{si} <{stage["name"]}> for film <{film}> succeeded in {diff_time}')

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filmfile", type=str, required=True,
                        help='path to the film file including filmID, such as 125261-name.mp4')
    parser.add_argument("--out-path", type=str, default=".",
                        help="directory where film-specific sub-directories are created")
    args = parser.parse_args()

    exit(0 if main(args) else 1)
    
