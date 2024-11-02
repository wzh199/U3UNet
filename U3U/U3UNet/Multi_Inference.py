import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

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
import shutil
from models import *


if __name__ == "__main__":
    print("gpu nums in current device: ", torch.cuda.device_count())
    torch.cuda.set_device(1)
    print("current gpu is: ", torch.cuda.current_device())

    dataset_path="./diy_video_to_image"  #Your dataset path
    model_path="../saved_models/permanent-saved/ite_321000_valLoss_1.2674_valTarLoss_0.1652_maxF1_0.6595_mae_0.0339_time_0.018236.pth"  # the model path
    # result_path="./diy_video_to_image_result"  #The folder path that you want to save the results
    input_size=[1024,1024]
    net=U3U515SimpleLayer()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net=net.cuda()
    else:
        net.load_state_dict(torch.load(model_path,map_location="cpu"))
    net.eval()
    im_list = glob(dataset_path+"/*.jpg")+glob(dataset_path+"/*.JPG")+glob(dataset_path+"/*.jpeg")+glob(dataset_path+"/*.JPEG")+glob(dataset_path+"/*.png")+glob(dataset_path+"/*.PNG")+glob(dataset_path+"/*.bmp")+glob(dataset_path+"/*.BMP")+glob(dataset_path+"/*.tiff")+glob(dataset_path+"/*.TIFF")
    print("im_list size:" + str(len(im_list)))
    record_imgs_total_gt0 = np.zeros(shape = len(im_list))
    for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):

        print("im_path: ", im_path)
        im = io.imread(im_path)
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        im_shp=im.shape[0:2]
        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
        im_tensor = F.upsample(torch.unsqueeze(im_tensor,0), input_size, mode="bilinear").type(torch.uint8)
        image = torch.divide(im_tensor,255.0)
        image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])

        if torch.cuda.is_available():
            image=image.cuda()

        with torch.no_grad():
            result=net(image)
        # print(result[0][0]) # TODO: validate what result is.
        result=torch.squeeze(F.upsample(result[0][0],im_shp,mode='bilinear'),0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result-mi)/(ma-mi)
        # result = result * 255

        # im_name=im_path.split('/')[-1].split('.')[0]
        matrix = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
        print(matrix[np.where(matrix != 0)].size)
        record_imgs_total_gt0[i] = matrix[np.where(matrix != 0)].size
        # io.imsave(os.path.join(result_path,im_name+".png"),(result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8))

    print(record_imgs_total_gt0)

# ///////////////////////////////////////////////////////////////////////////////
    model_path = "../saved_models/IS-Net-Fire"
    model_path_save = "../saved_models/Infe"
    pth_list = sorted(glob(model_path + "/*.pth"), key=lambda name: int(name[36:-4]))
    print(pth_list)

    dataset_path="./diy_video_to_image"  #Your dataset path
    # model_path="../saved_models/Infe/ite_261000_valLoss_2.7411_valTarLoss_0.4168_maxF1_0.6671_mae_0.0454_time_0.017936.pth"  # the model path
    # result_path="./diy_video_to_image_result"  #The folder path that you want to save the results
    input_size=[1024,1024]

    im_list = glob(dataset_path+"/*.jpg")+glob(dataset_path+"/*.JPG")+glob(dataset_path+"/*.jpeg")+glob(dataset_path+"/*.JPEG")+glob(dataset_path+"/*.png")+glob(dataset_path+"/*.PNG")+glob(dataset_path+"/*.bmp")+glob(dataset_path+"/*.BMP")+glob(dataset_path+"/*.tiff")+glob(dataset_path+"/*.TIFF")

    for i, pth_path in tqdm(enumerate(pth_list), total=len(pth_list)):
        print("pth_path: ", pth_path)
        print("i: ", str(i), str((i+1)*1000))

        net = U3U515BiasLayer()

        if torch.cuda.is_available():
            if (pth_path != ""):
                print("restore model from:")
                print(pth_path)
            net.load_state_dict(torch.load(pth_path))
            net = net.cuda()
        else:
            net.load_state_dict(torch.load(pth_path, map_location="cpu"))
        net.eval()

        do_i_need_save_this_model = True

        for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):
            print("gpu: " ,torch.cuda.current_device())
            print("im_path: ", im_path)
            im = io.imread(im_path)
            if len(im.shape) < 3:
                im = im[:, :, np.newaxis]
            im_shp=im.shape[0:2]
            im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
            im_tensor = F.upsample(torch.unsqueeze(im_tensor,0), input_size, mode="bilinear").type(torch.uint8)
            image = torch.divide(im_tensor,255.0)
            image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])

            if torch.cuda.is_available():
                image=image.cuda()

            with torch.no_grad():
                result=net(image)
            print(result[0][0]) # TODO: validate what result is.
            result=torch.squeeze(F.upsample(result[0][0],im_shp,mode='bilinear'),0)
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result-mi)/(ma-mi)

            matrix = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
            print(matrix[np.where(matrix != 0)].size)
            if matrix[np.where(matrix != 0)].size < record_imgs_total_gt0[i]:
                do_i_need_save_this_model = False

        if do_i_need_save_this_model == True:
            shutil.copy(pth_path, model_path_save)

            # im_name=im_path.split('/')[-1].split('.')[0]
            # io.imsave(os.path.join(result_path,im_name+".png"),(result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8))
