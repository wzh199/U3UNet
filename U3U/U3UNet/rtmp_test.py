import base64
import os
import curses
import sys

# import keyboard
# import rel

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import time
import numpy as np
from skimage import io
import time
from glob import glob
from tqdm import tqdm

import torch, gc
import torch.nn as nnc
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import websocket
import requests
from PIL import Image
from models import *

import subprocess as sp
import cv2
import sys
import queue
import threading

finished_frame_queue = queue.Queue()


def inference():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is not None:
                im = frame
                if len(im.shape) < 3:
                    im = im[:, :, np.newaxis]
                im_shp = im.shape[0:2]
                im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
                im_tensor = F.upsample(torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear").type(torch.uint8)
                image = torch.divide(im_tensor, 255.0)
                image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

                if torch.cuda.is_available():
                    image = image.cuda()

                with torch.no_grad():
                    result = net(image)
                # print(result[0][0])  # TODO: validate what result is.
                result = torch.squeeze(F.upsample(result[0][0], im_shp, mode='bilinear'), 0)
                ma = torch.max(result)
                mi = torch.min(result)
                result = (result - mi) / (ma - mi)
                result_img = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
                # print(str("test:") + str(type(result_img)))

                io.imsave(os.path.join(result_path, "1.jpg"),
                          result_img)
                # print("result_img", result_img)
                # return result_img
                f = open(result_path + "/1.jpg", "rb")
                # finished_frame_queue.put(f)
                ls_f = base64.b64encode(f.read())
                ws1.send(str(ls_f))

                fire_area_number_of_pixel = np.sum(result_img > 0)
                ws2.send(str(fire_area_number_of_pixel))


frame_queue = queue.Queue()

rtmpUrl = "rtmp://150.158.137.155/u3u"
camera_path = 'rtmp://150.158.137.155/below'

# 获取摄像头参数
cap = cv2.VideoCapture(camera_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# print(fps, width, height)

# ffmpeg command
# command = ['ffmpeg',
#            '-y',
#            '-f', 'rawvideo',
#            '-vcodec', 'rawvideo',
#            '-pix_fmt', 'bgr24',
#            '-s', "{}x{}".format(width, height),
#            '-r', str(fps),
#            '-i', '-',
#            '-c:v', 'libx264',
#            '-pix_fmt', 'yuv420p',
#            '-preset', 'ultrafast',
#            '-f', 'flv',
#            rtmpUrl]
command = ["ffmpeg",
           "-y",
           "-f", "image2pipe",
           "-vcodec", "mjpeg",
           "-r", "20",  # 输入帧率
           "-i", "-",  # 表示从标准输入读取数据
           # "-c:v", "libx264",
           "-pix_fmt", "yuv420p",
           # "-preset", "ultrafast",
           "-f", "flv",
           rtmpUrl]  # 输出的rtmp地址


# 读流函数
def Video():
    vid = cv2.VideoCapture(camera_path)
    if not vid.isOpened():
        raise IOError("could't open webcamera or video")
    while (vid.isOpened()):
        ret, frame = vid.read()
        # 下面注释的代码是为了防止摄像头打不开而造成断流
        # if not ret:
        # vid = cv2.VideoCapture(camera_path）
        # if not vid.isOpened():
        # raise IOError("couldn't open webcamera or video")
        # continue

        # f = inference(frame)
        # f = open(result_path + "/1.jpg", "rb")
        # print("f", f.read())

        frame_queue.put(frame)
        # finished_frame_queue.put(frame)


def push_stream():
    # 管道配置
    while True:
        if len(command) > 0:
            p = sp.Popen(command, stdin=sp.PIPE)
            break

    while True:
        if not finished_frame_queue.empty():
            frame = finished_frame_queue.get()
            if frame is not None:
                p.stdin.write(frame.read())
                # p.stdin.write(frame.tostring())


def run():
    thread_video = threading.Thread(target=Video, )
    thread_push = threading.Thread(target=push_stream, )
    thread_inference = threading.Thread(target=inference, )
    thread_video.start()
    thread_push.start()
    thread_inference.start()


if __name__ == "__main__":
    model_path = "/content/U3U/saved_models/Infe/ite_174000_valLoss_0.3175_valTarLoss_0.0442_maxF1_0.8513_mae_0.0142_time_0.016623.pth"  # the model path
    result_path = "/content/U3U/U3U-Net/test"  # The folder path that you want to save the results
    input_size = [1024, 1024]
    net = U3U515BiasLayer()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path, "cuda:0"))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    basic_url = "150.158.137.155:8080"

    websocket.enableTrace(True)

    ws1 = websocket.WebSocket()
    ws1.connect("ws://" + basic_url + "/webSocketFireAreaImage/1/fireAreaImage")

    ws2 = websocket.WebSocket()
    ws2.connect("ws://" + basic_url + "/webSocketCalcResult/1/1")

    run()



