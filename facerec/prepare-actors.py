#! /usr/bin/env python3

import re
import os
import argparse
import random
import zipfile
import requests
import tempfile
import cv2
import time
import json
from SPARQLWrapper import SPARQLWrapper, JSON

adir = '/u/18/jormal/unix/doc/projects/momaf/actors'
zipf = adir+'/actor-images.zip'

sparql_url = 'http://momaf-data.utu.fi:3030/momaf-raw/sparql'

film_sq = '''
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX momaf: <http://momaf-data.utu.fi/>
    
SELECT ?filmURI ?filmID ?filmname ?actorURI ?actorID (sample(?a) as ?actorname)
WHERE {
  ?filmURI a momaf:Movie ; 
          momaf:elonet_movie_ID <FILM>, ?filmID ;
          skos:prefLabel ?filmname ;
          momaf:hasMember [ 
            a momaf:Actor ;
            momaf:hasAgent ?actorURI
          ] .
  ?actorURI a momaf:Person ; 
          momaf:elonet_person_ID ?actorID ;
          skos:prefLabel ?a .
} GROUP BY ?filmURI ?filmID ?filmname ?actorURI ?actorID
'''

def fetch_actor_list(film):
    if type(film)==str:
        m = re.search('(\d+)', film)
        assert m, f'No numbers in film name <{film}>'
        film = f'"{m.group(1)}"'
    elif type(film)==int:
        film = f'"{film}"'
    assert type(film)==str,\
        f'fetch_actor_image_urls({film}) argument should be int or str'

    sparql = SPARQLWrapper(sparql_url)
    q = film_sq.replace('<FILM>', film)
    #print(q)
    sparql.setQuery(q)
    sparql.setReturnFormat(JSON)
    qresult = sparql.query()   # print(qresult.info())
    results = qresult.convert()

    l = []
    kk = [ a+b for a in ['film', 'actor'] for b in ['URI', 'ID', 'name'] ]
    for res in results['results']['bindings']:
        v = { k : res[k]['value'] if k in res else None for k in kk }
        l.append(v)
    return l

actor_sq = '''
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX momaf: <http://momaf-data.utu.fi/>
    
SELECT ?actorURI ?actorID (sample(?a) as ?actorname)
       ?imageURI ?image_url ?filename ?filmURI ?filmID ?filmname

WHERE {
  ?actorURI a momaf:Person ; 
          momaf:elonet_person_ID <ACTOR>, ?actorID ;
          skos:prefLabel ?a .
  ?imageURI a momaf:Image ; 
          momaf:hasMember [ momaf:hasAgent ?actorURI ] ;
          momaf:sourcefile ?image_url ;
          skos:prefLabel ?filename ;                     # or momaf:elonet_ID ???
          momaf:hasMember [ momaf:hasAgent ?filmURI ] . 
  ?filmURI a momaf:Movie ; 
          momaf:elonet_movie_ID <FILM> , ?filmID ;
          skos:prefLabel ?filmname .
} GROUP BY ?filmURI ?filmID ?filmname ?actorURI ?actorID ?imageURI ?image_url ?filename
'''

def fetch_actor_image_urls(actor, film):
    if type(actor)==str:
        m = re.search('(\d+)', actor)
        assert m, f'No numbers in actor name <{actor}>'
        actor = f'"{m.group(1)}"'
    elif type(actor)==int:
        actor = f'"{actor}"'
    assert type(actor)==str,\
        f'fetch_actor_image_urls({actor},*) argument should be int or str'

    if type(film)==str:
        m = re.search('(\d+)', film)
        assert m, f'No numbers in film name <{film}>'
        film = f'"{m.group(1)}"'
    elif type(film)==int:
        film = f'"{film}"'
    elif film is None:
        film = '?film'
    assert type(film)==str,\
        f'fetch_actor_image_urls(*,{film}) argument should be int or str'

    sparql = SPARQLWrapper(sparql_url)
    q = actor_sq.replace('<ACTOR>', actor).replace('<FILM>', film)
    #print(q)
    sparql.setQuery(q)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    l = []
    kk = [ a+b for a in ['film', 'actor'] for b in ['URI', 'ID', 'name'] ]
    kk.extend(['imageURI', 'image_url', 'filename'])
    for res in results['results']['bindings']:
        v = { k : res[k]['value'] if k in res else None for k in kk }
        l.append(v)
    return l


def select_image_set(film, nimg):
    """ Not used... """
    
    alist = fetch_actor_list(film)
    ilist = []
    for a in alist:
        aid, aname = (a["actorID"], a["actorname"])
        l1 = fetch_actor_image_urls(aid, film)
        ll1 = len(l1)
        l2 = []
        ll2 = len(l2)
        if ll1>nimg:
            l1 = l1[:nimg]
        elif ll1<nimg:
            l2 = fetch_actor_image_urls(aid, None)
            random.shuffle(l2)
            ll2 = len(l2)
            if ll2>nimg-ll1:
                l2 = l2[:nimg-ll1]
        print(f'{aid} {aname} {ll1} -> {len(l1)} + {ll2} -> {len(l2)}')
            
        ilist.extend(l1)
        ilist.extend(l2)
    return ilist


def fetch_and_store_image(iurl, aimg, iname):
    r = requests.get(iurl)
    if r.status_code!=200:
        print(f'FAILED to retrieve {iurl} : {r.status_code}')
        return None
    aimg.writestr(iname, r.content)
    return r.content


facenet_models = [ '20180402-114759', '20180408-102900',
                   '20170511-185253', '20170512-110547' ]
detector = None
embedders = {}
def detect_and_embed_face(idata, iname):
    from extract import bbox_float_to_int
    from detector import FaceNetDetector
    from keras_facenet.utils import cropBox
    from keras_facenet import FaceNet

    global detector, embedders
    if detector is None:
        detector = FaceNetDetector(min_face_size=20, face_threshold=0.95)
        embedders = { i : FaceNet(key=i) for i in facenet_models }
        
    tfile = tempfile.NamedTemporaryFile(suffix='.jpeg', buffering=0)
    tfile.write(idata)
    #print(tfile.name, iname, len(idata), os.path.getsize(tfile.name))
    # time.sleep(3)
    img = cv2.imread(tfile.name)
    # print(img)

    faces = detector.detect(img)
    if len(faces)!=1:
        return None

    d_height, d_width = img.shape[:2]
    print(iname, d_width, d_height, faces[0])
    tight_box = bbox_float_to_int(faces[0]['box'], d_width, d_height)
    det = { 'box': [tight_box[0], tight_box[1],
                    tight_box[2]-tight_box[0], tight_box[3]-tight_box[1]] }
    margin = int(0.1*160)
    cropped = cropBox(img, detection=det, margin=margin)
    #print(cropped)
    embeddings = { i : embedders[i].embeddings([cropped])[0].tolist()
                   for i in embedders.keys() }

    
    return {'box': tight_box, 'embeddings': embeddings}


def prepare_one_actor(a, nimg):
    files_in_zip = []
    if os.path.isfile(zipf):
        with zipfile.ZipFile(zipf) as aimg:    
            files_in_zip = aimg.namelist()
    if not os.path.exists(adir):
        os.mkdir(adir)
    aimg = zipfile.ZipFile(zipf, 'a')

    fid, aid, aname = (a['filmID'], a['actorID'], a['actorname'])
    l = fetch_actor_image_urls(aid, None)
    faces = []
    for m in (True, False):
        for i in l:
            ifid, ifname, iurl, iname = (i['filmID'], i['filmname'], i['image_url'], i['filename'])
            if m==(ifid==fid):
                #print(m, i)
                jname = iname+'.json'
                j = jname in files_in_zip
                e = iname in files_in_zip
                print(m, e, j, aname, ifid, iurl, iname, jname)
                if not e:
                    idata = fetch_and_store_image(iurl, aimg, iname)
                    if idata is None:
                        continue
                    files_in_zip.append(iname)
                elif not j:
                    idata = aimg.read(iname)
                if not j:
                    # print(f'  {iname} {e} {len(idata)}')
                    face = detect_and_embed_face(idata, iname)
                    if face is None:
                        face = { 'note': 'no unique face' }
                    else:
                        face['actorID']   = aid
                        face['actorname'] = aname
                    face['filmID']    = ifid
                    face['filmname']  = ifname
                    face['image_url'] = iurl
                    face['filename']  = iname
                    jdata = json.dumps(face)
                    aimg.writestr(jname, jdata)
                    files_in_zip.append(jname)
                else:
                    face = json.loads(aimg.read(jname))
                if 'box' in face:
                    faces.append(face)
            if len(faces)>=nimg:
                break
        if len(faces)>=nimg:
            break
    return faces

        
if __name__=='__main__':
    parser = argparse.ArgumentParser(allow_abbrev=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Utility for collecting actor face embeddings for a film.')
    parser.add_argument('--film', type=str, required=True, help='filmID, such as 125261-name-of-the-movie')
    parser.add_argument('--n-faces', type=int, default=20, help='number of faces per actor')
    parser.add_argument('--path', type=str, default='.',
                        help='path to JSON data directory for a film')
    args = parser.parse_args()

    # select_image_set(args.film, args.n_faces)

    faces = []
    alist = fetch_actor_list(args.film)
    if len(alist)==0:
        print(f'No actors found for film <{args.film}>')
        exit(1)
    
    for a in alist:
        faces.extend(prepare_one_actor(a, args.n_faces))

    if len(faces)==0:
        print(f'No actor faces found for film <{args.film}>')
        exit(1)

    json.dump(faces, open(args.path+'/actor-faces-'+alist[0]['filmID']+'.json', 'w'))
