# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion : 
    @Author    :
    @Time      :
    @Detail    :
'''

# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import torch
import argparse

from PIL import Image, ImageDraw

from os.path import join, dirname, abspath, isfile
CURRENT_DIR = dirname(abspath(__file__))

"""hyper parameters"""
use_cuda = True

def detect_pil(cfgfile, weightfile, img):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = CURRENT_DIR + '/data/voc.names'
    elif num_classes == 80:
        namesfile = CURRENT_DIR + '/data/coco.names'
    else:
        namesfile = CURRENT_DIR + '/data/x.names'
    class_names = load_class_names(namesfile)

    sized = img.resize((m.width, m.height))
    sized_np = np.array(sized)
    # img = cv2.imread(imgfile)
    # sized = cv2.resize(img, (m.width, m.height))
    # sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized_np, 0.4, 0.6, use_cuda)
        finish = time.time()
        if i == 1:
            print('Predicted in %f seconds.' % (finish - start))

    # sized.show()

    # save_name = CURRENT_DIR + "/prediction.jpg"
    # sized.save(save_name)

    plot_boxes_pil(img, boxes[0], savename= CURRENT_DIR + "/prediction.jpg", class_names=class_names)


if __name__ == '__main__':
    cfgfile = CURRENT_DIR + '/cfg/yolov4.cfg'
    weightfile = CURRENT_DIR + '/weights/yolov4.weights'
    imgfile = CURRENT_DIR + '/data/dog.jpg'
    img = Image.open(imgfile)

    print(weightfile)

    detect_pil(cfgfile, weightfile, img)
        
