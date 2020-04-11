#! /usr/bin/env python3

import mtcnn
#import imageio
import face_utils
import numpy as np
import utils
import os
import sys
import tensorflow
from PIL import Image, ImageDraw
import cv2

margin = 0
image_size = 160
min_face_size = 80

facenet = tensorflow.keras.models.load_model('facenet_keras.h5')
facenet.load_weights('facenet_keras_weights.h5')
detector = mtcnn.MTCNN(min_face_size=min_face_size)


def process_image(file):
    label,_ = os.path.splitext(os.path.basename(file))
    img = Image.open(file).convert('RGB')
    return process_frame(np.asarray(img), label)


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / float(boxAArea + boxBArea - interArea)


def process(buf, iou_thr=0.5):
    debug = False;
    
    midx  = len(buf)//2
    mid   = buf[midx].copy()
    boxes = mid['boxes'].copy()
    mid['boxes'] = []

    for b in boxes:
        bx = b['box']
        if debug:
            print(mid['label'], bx)
        keep = True
        for i in range(len(buf)):
            if i==midx:
                continue
            found = False
            for c in buf[i]['boxes']:
                cx = c['box']
                fx = ''
                iu = iou(bx, cx)
                if iu>iou_thr:
                    found = True
                    fx = 'match'
                if debug:
                    print(' ', i, buf[i]['label'], cx, iu, fx)
                if found:
                    break
            if not found:
                keep = False
                break
        if debug:
            print(' ', 'found' if keep else 'not found')
        if keep:
           mid['boxes'].append(b) 
            
    if debug:
        print()
    
    return mid


def process_video(file, beg = 0, end = -1, margin = 2, iou = 0.9):
    debug = False
    
    label,_ = os.path.splitext(os.path.basename(file))
    cap = cv2.VideoCapture(file)
    buf = []
    f = 0
    while True:
        skip = f<beg-margin
        proc = len(buf)==2*margin
        stop = end>=0 and f>=end+margin
        msg = 'skipping' if skip else 'stopping' if stop else \
            'processing' if proc else 'buffering'
        flabel = label+':'+str(f)
        f += 1
        if debug:
            print(flabel, msg)
        if stop:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if skip:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ff = process_frame(rgb, flabel)
        ff['frame'] = f-1
        #save_images(ff, False, True)
        print_features(ff)
        buf.append(ff)
        if len(buf)==2*margin+1:
            fx = process(buf)
            for b in fx['boxes']:
                b['label'] = 'kept-'+b['label']
            save_images(fx, True, True)
            print_features(fx)
            buf = buf[1:]
        
    cap.release()


def process_frame(npimg, label_f):
    img_shape = npimg.shape
    bbs = detector.detect_faces(npimg)
    det = np.array([utils.fix_box(b['box']) for b in bbs])
    det_arr = []
    for i in range(len(bbs)):
        det_arr.append(np.squeeze(det[i]))

    img  = Image.fromarray(npimg)
    imgb = img.copy()
    draw = ImageDraw.Draw(imgb)

    ret = { 'label': label_f, 'image': img, 'boxed': imgb }
    arr = []
    
    for i, det in enumerate(det_arr):
        info = {}
        det = np.squeeze(utils.xywh2rect(*det))
        draw.rectangle(det.tolist(), fill=None, outline=None)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_shape[0])
        cropped = img.crop((bb[0], bb[1], bb[2], bb[3]))
        scaled = np.array(cropped
                          .resize((image_size, image_size),
                                  resample=Image.BILINEAR)
                          .convert('L'))
        label = label_f+'_{}_{}_{}_{}'.format(bb[0], bb[1], bb[2], bb[3])
        scaledx = np.array(cropped
                          .resize((image_size, image_size),
                                  resample=Image.BILINEAR))
        scaledx = scaledx.reshape(-1, image_size, image_size, 3)
        emb = utils.get_embedding(facenet, scaledx[0])

        info['box']      = bb 
        info['crop']     = cropped
        info['scaled']   = scaled
        info['label']    = label 
        info['features'] = emb
        arr.append(info)

    ret['boxes'] = arr
    return ret
        

def print_features(ff):
    for b in ff['boxes']:
        print(' '.join([str(x) for x in b['features']]), b['label'])


def save_images(ff, s, z):
    if s:
        for b in ff['boxes']:
            Image.fromarray(b['scaled']).save(b['label']+'.jpeg')
    if z:
        ff['boxed'].save(ff['label']+'_boxed.jpeg')

        
if __name__ == "__main__":
    a = sys.argv
    _, ext = os.path.splitext(os.path.basename(a[1]))
    if ext in ['.jpg', '.jpeg', '.png']:
        for f in a[1:]:
            ff = process_image(f)
            # print(ff)
            print_features(ff)
            save_images(ff, True, True)
    if ext in ['.mpeg', '.mpg', '.mp4', '.avi', '.wmv']:
        if len(a)!=4:
            print(a[0],
                  ': three arguments needed: file.mp4 begin_frame end_frame')
            exit(1)
        ff = process_video(a[1], beg=int(a[2]), end=int(a[3]))
        
