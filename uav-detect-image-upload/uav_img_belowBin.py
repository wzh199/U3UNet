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

def belowbincallback(rgbimage):
    bridge = CvBridge()
    global belowbinframe
    belowbinframe = bridge.imgmsg_to_cv2(rgbimage, "mono8")
    # print(frame.shape)
    # cv2.imshow("belowbin", belowbinframe)
    # cv2.waitKey(1)
    cv2.imwrite("belowbinframe.jpg", belowbinframe)

    f = open("belowbinframe.jpg", "rb")
    ls_f = base64.b64encode(f.read())
    ws.send(str(ls_f))
    # url = 'http://127.0.0.1:8080/upload3'
    # data = {"type": "belowBinary"}
    # files = {
    #     "file": open("belowbinframe.jpg", "rb"),
    # }
    # r = session.post(url, data, files=files)
    # print(r.text)


def listener():
    rospy.init_node('upload_3', anonymous=True)
    rospy.Subscriber('/below/camera/bin', Image, belowbincallback)
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
    ws.connect("ws://"+ basic_url +"/webSocket3/1/belowBinary")

    listener()
    ws.close()