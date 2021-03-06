from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

def run(opt, image_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    img_folder = os.path.join(image_folder, 'images')
    dataloader = DataLoader(
        ImageFolder(img_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    ## TODO : make class id assigned autometically from rolo images
    target_cls = []
    with open(os.path.join(image_folder,'class.txt'), 'r') as file:
        try:
            for n in file.readlines():
                target_cls.append(int(n))
        except ValueError:
            print(target_cls)

    gt_list = []
    with open(os.path.join(image_folder,'groundtruth_rect.txt'), 'r') as file:
        labels = file.readlines()
        
    for label in labels:
        l = label.split('\t')   # for gt type 2
        if len(l) < 4:
            l = label.split(',') # for gt type 1
        gt_list.append(l)

    # Saving folder
    foldername = os.path.join(image_folder,'yot_out')
    if not os.path.exists(foldername):
        os.makedirs(foldername)
        
    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            (detections, colimg) = model(input_imgs)
            detection_list = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
                        
        for detections in detection_list:
            img = np.array(Image.open(img_paths[0]))
            # Find a location with the selected class ID
            location = np.array([0.5,0.5,0,0,0], dtype=float) #np.zeros(5, dtype=float)

            if detections is not None:
                # Rescale boxes to original image
                detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                
                max_iou = 0 #opt.tracking_thres

                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    if int(cls_pred) not in target_cls:
                        print("\t\tclass id :", cls_pred)
                        continue
                    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                    
                    box_w = x2 - x1
                    box_h = y2 - y1
                    cx = x1 + box_w/2
                    cy = y1 + box_h/2

                    b1 = torch.tensor([[cx, cy, box_w, box_h]])
                    
                    b2 = np.array(gt_list[batch_i], dtype=float)
                    b2[0] += b2[2]/2.
                    b2[1] += b2[3]/2.
                    b2 = torch.tensor([b2], dtype=torch.float32)
                    
                    iou = bbox_iou(b1, b2, False)
                    if iou >= max_iou:
                        max_iou = iou

                        # Normalize the coordinates with image width and height and class id
                        # [cx, cy, width, heigt, confidence, class id]
                        location[0] = cx/img.shape[1]
                        location[1] = cy/img.shape[0]
                        location[2] = box_w/img.shape[1]
                        location[3] = box_h/img.shape[0]
                        location[4] = conf

        # save a location and a feature image
        filename = img_paths[0].split("/")[-1].split(".")[0]
        save = np.concatenate((colimg.reshape(-1).numpy(), location))
            
        np.save(f"{foldername}/{filename}.npy", save)
        print("\t Saving the location : ", save[-5:])
             
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

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
    parser.add_argument("--tracking_thres", type=float, default=0.2, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    f = opt.image_folder
    if f.lower().endswith(('*')):
        root = os.path.dirname(f)
        for l in os.listdir(root):
            l = os.path.join(root, l)
            run(opt, l)
            print(l)
    else:
        run(opt, f)
