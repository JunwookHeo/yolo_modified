from __future__ import division

from models import *
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from utils.utils import *
from utils.datasets import *

import argparse
import os 
import os.path as osp

import pickle as pkl
import pandas as pd
import random

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--video", type = str, default = "video2.mp4", dest = "videofile", help = "Video file to run detection on")
    opt = parser.parse_args()
    print(opt)

    return opt
    
args = arg_parse()
batch_size = int(args.batch_size)
confidence = float(args.conf_thres)
nms_thesh = float(args.nms_thres)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = load_classes(args.class_path)
num_classes = len(classes) #80

#Set up the neural network
print("Loading network.....")
model = Darknet(args.model_def, img_size=args.img_size).to(device)
if args.weights_path.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(args.weights_path)
else:
    # Load checkpoint weights
    model.load_state_dict(torch.load(args.weights_path))

model.eval()  # Set in evaluation mode

print("Network successfully loaded")

inp_dim = model.img_size
assert inp_dim % 32 == 0 
assert inp_dim > 32

#Set the model in evaluation mode
model.eval()

def write(x, results):
    c1 = tuple(x[0:2].int())
    c2 = tuple(x[2:4].int())
    img = results
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

#Detection phase

videofile = args.videofile #or path to the video file. 

cap = cv2.VideoCapture(videofile)  

#cap = cv2.VideoCapture(0)  for webcam

assert cap.isOpened(), 'Cannot capture source'

frames = 140
start = time.time()
cap.set(cv2.CAP_PROP_POS_FRAMES, frames)

while cap.isOpened():
    ret, frame = cap.read()

    if ret:   
        img = prep_image(frame, inp_dim)
        cv2.imwrite('video2_{:d}.jpg'.format(frames), frame)
#        cv2.imshow("a", frame)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1,2)   
                     
        if torch.cuda.is_available():
            im_dim = im_dim.cuda()
            img = img.cuda()
        
        with torch.no_grad():
            output = model(Variable(img))
            output = non_max_suppression(output, args.conf_thres, args.nms_thres)

        if output is None:
            frames += 1
            print("FPS of the video is {:5.4f}".format( frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue
        
        for o in output:
            if o is None:
                frames += 1
                print("FPS of the video is {:5.4f}".format( frames / (time.time() - start)))
                cv2.imshow("frame", frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            #o = rescale_boxes(o, args.img_size, img.shape[:2])
            
            im_dim = im_dim.repeat(o.size(0), 1)
            scaling_factor = torch.min(416/im_dim,1)[0].view(-1,1)
            
            o[:,[0,2]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
            o[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
            
            o[:,0:4] /= scaling_factor
            
            for i in range(o.shape[0]):
                o[i, [0,2]] = torch.clamp(o[i, [0,2]], 0.0, im_dim[i,0])
                o[i, [1,3]] = torch.clamp(o[i, [1,3]], 0.0, im_dim[i,1])
            
            #classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))

            list(map(lambda x: write(x, frame), o))
            
            cv2.imshow("frame", frame)
            #cv2.imwrite('video4{:03d}.png'.format(frames), frame)
        
            frames += 1
            print(time.time() - start)
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        
    else:
        break     






