# Generating yolo test list from rolo dataset.
import sys
import os
import numpy as np
from PIL import Image

if __name__ == "__main__":
    print(sys.argv)
    target_dir = '../yot_unseen'

    # Rename img to images
    for root, dirs,_ in os.walk(target_dir):
        for d in dirs:
            if d == 'img' and os.path.exists(os.path.join(root,'groundtruth_rect.txt')):
                s_path = os.path.join(root,d)
                d_path = os.path.join(root,'images')
                os.rename(s_path, d_path)
                
    # Create lables folder
    # Generate train.txt file and valid.txt including image list.
    # Generate label files
    for root, dirs,_ in os.walk(target_dir):
        for d in dirs:
            if d == 'images' and os.path.exists(os.path.join(root,'groundtruth_rect.txt')):
                lables = os.path.join(root,'labels')
                if not os.path.exists(lables):                    
                    os.mkdir(lables)
                
                files = os.listdir(lables)
                for f in files:
                    os.remove(os.path.join(lables, f))

                images = os.path.join(root,d)
                print('Processing ' + images)
                files = os.listdir(images)

                # Generate train.txt
                with open(os.path.join(root,'train.txt'), "w") as file:
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg')):
                            file.write(os.path.abspath(os.path.join(images,f)) + '\n')

                # Gnerate valid.txt
                with open(os.path.join(root,'valid.txt'), "w") as file:
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg')):
                            file.write(os.path.abspath(os.path.join(images,f)) + '\n')

                # Gnerate label files
                target_cls = []
                with open(os.path.join(root,'class.txt'), 'r') as file:
                    try:
                        for n in file.readlines():
                            target_cls.append(int(n))
                    except ValueError:
                        print(target_cls)
                        
                name = os.path.join(root,'groundtruth_rect.txt')
                if os.path.exists(name):
                    try:
                        gts = np.loadtxt(name, delimiter=',')
                    except ValueError:
                        gts = np.loadtxt(name, delimiter='\t')

                for i, f in enumerate(files):                    
                    frame = np.array(Image.open(os.path.join(images, f)))
                    if f.lower().endswith(('.png', '.jpg')):
                        l = f.replace(".png", ".txt").replace(".jpg", ".txt")
                        with open(os.path.join(lables, l), "w") as file:
                            try:
                                gt = gts[i]
                                cx = gt[0]/frame.shape[1]
                                cy = gt[1]/frame.shape[0]
                                w = gt[2]/frame.shape[1]
                                h = gt[3]/frame.shape[0]
                                cx = cx + w/2
                                cy = cy + h/2
                                file.write("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(target_cls[0], cx, cy, w, h))
                            except:
                                print('Exception : ' + root + ' ' + f)


                '''
                label_list = []
                with open(os.path.join(root,'groundtruth_rect.txt')) as file:
                    label_list = file.readlines()
                
                for i, f in enumerate(files):
                    if f.lower().endswith(('.png', '.jpg')):
                        l = f.replace(".png", ".txt").replace(".jpg", ".txt")
                        with open(os.path.join(lables, l), "w") as file:
                            try:
                                file.write(label_list[i])
                            except:
                                print('Exception : ' + root + ' ' + f)

                '''