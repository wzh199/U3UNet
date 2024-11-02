#!/usr/bin/env python3.8
# [image upload to web server]
import base64
import websocket
import requests
import message_filters
import numpy as np
import rospy  # 导入rospy功能包
from cv_bridge import CvBridge
import cv2
import sensor_msgs.msg
from std_msgs.msg import Float32  # 导入std_msgs/Float32消息功能包
from sensor_msgs.msg import CompressedImage, Image
from urllib.parse import urlencode

global frontframe
frontframe = None

global belowframe
belowframe = None

global belowbinframe
belowbinframe = None

session_ = requests.session()

def frontcallback(rgbimage):
    print("callback")
    bridge = CvBridge()
    global frontframe
    frontframe = bridge.imgmsg_to_cv2(rgbimage, "bgr8")
    # cv2.imshow("front", frontframe)
    # cv2.waitKey(1)
    # print(frame.shape)
    cv2.imwrite("frontframe.jpg", frontframe)

    f = open("frontframe.jpg", "rb")
    ls_f = base64.b64encode(f.read())
    ws.send(str(ls_f))
    # url = 'http://127.0.0.1:8080/upload'
    # data = {"type": "front"}
    # files = {
    #     "file": open("frontframe.jpg", "rb"),
    # }
    # r = session_.post(url, data, files=files)
    # print(r.text)

def listener():
    rospy.init_node('upload_1', anonymous=True)
    rospy.Subscriber('/front/camera/image', Image, frontcallback)
    # rospy.Subscriber('/airsim/camera_2/compressed/rgb/image_rect_color/compressed', CompressedImage, belowcallback)
    # rospy.Subscriber('/below/camera/bin', Image, belowbincallback)
    # rospy.Subscriber("/unreal_ros/image_color2", Image, callback)

    rate = rospy.Rate(10)

    while (not rospy.is_shutdown()):


        rate.sleep()

    rospy.spin()


if __name__ == '__main__':

    f = open("basic_url.txt", "r")
    basic_url = f.read()
    f.close()

    requests.post('http://'+ basic_url +'/deleteUploadImages')

    ws = websocket.WebSocket()
    ws.connect("ws://"+ basic_url +"/webSocket/1/front")

    listener()
    ws.close()
