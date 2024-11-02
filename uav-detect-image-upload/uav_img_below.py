#!/usr/bin/env python3.8
# [image upload to web server]
import base64

import requests
import message_filters
import numpy as np
import rospy  # 导入rospy功能包
import websocket
from cv_bridge import CvBridge
import cv2
import sensor_msgs.msg
from std_msgs.msg import Float32  # 导入std_msgs/Float32消息功能包
from sensor_msgs.msg import CompressedImage,Image

global frontframe
frontframe = None

global belowframe
belowframe = None

global belowbinframe
belowbinframe = None


session = requests.session()


def belowcallback(rgbimage):
    print("callback")
    bridge = CvBridge()
    global belowframe
    belowframe = bridge.compressed_imgmsg_to_cv2(rgbimage, "bgr8")
    # print(frame.shape)
    # cv2.imshow("below", belowframe)
    # cv2.waitKey(1)
    cv2.imwrite("belowframe.jpg", belowframe)


    f = open("belowframe.jpg", "rb")
    ls_f = base64.b64encode(f.read())
    ws.send(str(ls_f))
    # ws1.send(str(ls_f))
    # url = 'http://127.0.0.1:8080/upload2'
    # data = {"type": "belowRGB"}
    # files = {
    #     "file": open("belowframe.jpg", "rb"),
    # }
    # r = session.post(url, data, files=files)
    # print(r.text)


def listener():
    rospy.init_node('upload_2', anonymous=True)
    rospy.Subscriber('/airsim/camera_2/compressed/rgb/image_rect_color/compressed', CompressedImage, belowcallback)
    # rospy.Subscriber("/unreal_ros/image_color2", Image, callback)
    rate = rospy.Rate(10)
    while (not rospy.is_shutdown()):

        rate.sleep()

    rospy.spin()

if __name__ == '__main__':

    f = open("basic_url.txt", "r")
    basic_url = f.read()
    f.close()

    ws = websocket.WebSocket()
    ws.connect("ws://"+ basic_url +"/webSocket2/1/belowRGB")
    # ws1 = websocket.WebSocket()
    # ws1.connect("ws://"+ "150.158.137.155:8080" +"/webSocket2/1/belowRGB")

    listener()
    ws.close()
    # ws1.close()
