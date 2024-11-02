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

global count
count = 10

session = requests.session()


def belowcallback(rgbimage):
    print("callback")
    bridge = CvBridge()
    global belowframe
    belowframe = bridge.compressed_imgmsg_to_cv2(rgbimage, "bgr8")
    # print(frame.shape)
    # cv2.imshow("below", belowframe)
    # cv2.waitKey(1)
    global count
    count -= 1
    print("count:", count)
    if count == 0:
        cv2.imwrite("belowframe_server.jpg", belowframe)
        f = open("belowframe_server.jpg", "rb")
        ls_f = base64.b64encode(f.read())
        ws1.send(str(ls_f))
        count = 10




def listener():
    rospy.init_node('upload_4', anonymous=True)
    rospy.Subscriber('/airsim/camera_2/compressed/rgb/image_rect_color/compressed', CompressedImage, belowcallback)
    # rospy.Subscriber("/unreal_ros/image_color2", Image, callback)
    rate = rospy.Rate(30)
    while (not rospy.is_shutdown()):

        rate.sleep()

    rospy.spin()


def on_message(ws, message):
    print("on_message:" + message)
    if message == "无人机火焰捕获":
        ws.close()

def on_error(ws, error):
    print(error)

def on_close(ws, close_status_code, close_msg):
    print("### closed ###")

def on_open(ws):
    print("Opened connection")

if __name__ == '__main__':

    f = open("basic_url.txt", "r")
    basic_url = f.read()
    f.close()

    # ws0 = websocket.WebSocket()
    # ws0.connect("ws://"+ basic_url +"/UAVHeightWebSocket/3/1")
    ws1 = websocket.WebSocket()
    ws1.connect("ws://"+ "150.158.137.155:8080" +"/webSocket2/1/belowRGB")


    ws = websocket.WebSocketApp("ws://"+ basic_url +"/UAVHeightWebSocket/3/1",
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    ws.run_forever()

    listener()
    # ws0.close()
    ws1.close()
