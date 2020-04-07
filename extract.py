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

margin = 0
image_size = 160
min_face_size = 120

facenet = tensorflow.keras.models.load_model('facenet_keras.h5')
facenet.load_weights('facenet_keras_weights.h5')

detector = mtcnn.MTCNN(min_face_size=min_face_size)

for f in sys.argv[1:]:
    # img = imageio.imread(f)
    # if img.ndim == 2:
    #     img = facenet.to_rgb(img)
    # img = img[:, :, 0:3]
    img = Image.open(f).convert('RGB')
    img_size = img.size

    bbs = detector.detect_faces(np.asarray(img))
    det = np.array([utils.fix_box(b['box']) for b in bbs])
    det_arr = []
    for i in range(len(bbs)):
        det_arr.append(np.squeeze(det[i]))

    imgb = img
    draw = ImageDraw.Draw(imgb)
  
    label_f, file_extension = os.path.splitext(os.path.basename(f))

    for i, det in enumerate(det_arr):
        det = np.squeeze(utils.xywh2rect(*det))
        draw.rectangle(det.tolist(), fill=None, outline=None)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[0])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[1])
        cropped = img.crop((bb[0], bb[1], bb[2], bb[3]))
        scaled = np.array(cropped
                          .resize((image_size, image_size), resample=Image.BILINEAR)
                          .convert('L'))
        label = label_f+'_{}_{}_{}_{}'.format(bb[0], bb[1], bb[2], bb[3])
        Image.fromarray(scaled).save(label+file_extension)

        scaledx = np.array(cropped
                          .resize((image_size, image_size), resample=Image.BILINEAR))
        scaledx = scaledx.reshape(-1, image_size, image_size, 3)
        emb_array = [utils.get_embedding(facenet, face_pixels) for face_pixels in scaledx]
        emb_array = np.asarray(emb_array)

        for v in emb_array[0]:
            print(v, end=' ')
        print(label)

    if len(det_arr):
        imgb.save(label_f+'_boxed.png')
