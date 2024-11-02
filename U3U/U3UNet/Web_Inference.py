import base64
import os
import curses
import sys

import keyboard
#import rel

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import time
import numpy as np
from skimage import io
import time
from glob import glob
from tqdm import tqdm

import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import websocket
import requests
from PIL import Image

from models import *


def inference(img_np):
    im = io.imread(img_np)
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
    print(result[0][0])  # TODO: validate what result is.
    result = torch.squeeze(F.upsample(result[0][0], im_shp, mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    result_img = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    # print(str("test:") + str(type(result_img)))
    io.imsave(os.path.join(result_path, "1.jpg"),
              result_img)

    f = open(result_path + "/1.jpg", "rb")
    ls_f = base64.b64encode(f.read())

    fire_area_number_of_pixel = np.sum(result_img > 0)

    ws1.send(str(ls_f))
    ws2.send(str(fire_area_number_of_pixel))




def on_message(ws, message):
    print(message)
    url = "http://" + basic_url + "/" + message
    print(url)
    # r = requests.get(url)
    # img_pil = Image.open(r.raw)
    # img_np = np.array(img_pil)
    inference(url)


def on_error(ws, error):
    print(error)


def on_close(ws, close_status_code, close_msg):
    print("### closed ###")


def on_open(ws):
    print("Opened connection")


if __name__ == "__main__":
    print("gpu nums: ", torch.cuda.device_count())
    #torch.cuda.set_device(1)
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

    # basic_url = "127.0.0.1:8080"
    basic_url = "150.158.137.155:8080"

    websocket.enableTrace(True)

    ws1 = websocket.WebSocket()
    ws1.connect("ws://" + basic_url + "/webSocketFireAreaImage/1/fireAreaImage")

    ws2 = websocket.WebSocket()
    ws2.connect("ws://" + basic_url + "/webSocketCalcResult/1/1")

    ws = websocket.WebSocketApp("ws://" + basic_url + "/webSocket2/3/belowRGB",
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()  # Set dispatcher to automatic reconnection, 5 second reconnect delay if connection closed unexpectedly

    ws1.close()
    ws2.close()


