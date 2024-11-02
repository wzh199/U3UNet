import rospy
from sensor_msgs.msg import Image,CameraInfo
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
from std_msgs.msg import String
import airsim
import cv2
import numpy as np
import websocket

CLAHE_ENABLED = False  # when enabled, RGB image is enhanced using CLAHE

CAMERA_FX = 320
CAMERA_FY = 320
CAMERA_CX = 320
CAMERA_CY = 240

CAMERA_K1 = -0.000591
CAMERA_K2 = 0.000519
CAMERA_P1 = 0.000001
CAMERA_P2 = -0.000030
CAMERA_P3 = 0.0

IMAGE_WIDTH = 640  # resolution should match values in settings.json
IMAGE_HEIGHT = 480


class KinectPublisher:
    def __init__(self):
        self.bridge_rgb = CvBridge()
        self.msg_rgb = Image()
        self.bridge_d = CvBridge()
        self.msg_d = Image()
        self.msg_info = CameraInfo()
        self.msg_tf = TFMessage()

    def getDepthImage(self,response_d):
        img_depth = np.array(response_d.image_data_float, dtype=np.float32)
        img_depth = img_depth.reshape(response_d.height, response_d.width)
        return img_depth

    def getRGBImage(self,response_rgb):
        img1d = np.fromstring(response_rgb.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response_rgb.height, response_rgb.width, 3)
        # img_rgb = img_rgb[..., :3][..., ::-1]
        return img_rgb

    def enhanceRGB(self,img_rgb):
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        lab_planes_list = list(lab_planes) #
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(10, 10))
        lab_planes_list[0] = clahe.apply(lab_planes_list[0]) #
        lab_planes_list = tuple(lab_planes_list) #
        lab = cv2.merge(lab_planes)
        img_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return img_rgb

    def GetCurrentTime(self):
        self.ros_time = rospy.Time.now()

    def CreateRGBMessage(self,img_rgb):
        self.msg_rgb.header.stamp = self.ros_time
        self.msg_rgb.header.frame_id = "camera_rgb_optical_frame"
        self.msg_rgb.encoding = "bgr8"
        self.msg_rgb.height = IMAGE_HEIGHT
        self.msg_rgb.width = IMAGE_WIDTH
        self.msg_rgb.data = self.bridge_rgb.cv2_to_imgmsg(img_rgb, "bgr8").data
        self.msg_rgb.is_bigendian = 0
        self.msg_rgb.step = self.msg_rgb.width * 3
        return self.msg_rgb

    def CreateDMessage(self,img_depth):
        self.msg_d.header.stamp = self.ros_time
        self.msg_d.header.frame_id = "camera_depth_optical_frame"
        self.msg_d.encoding = "32FC1"
        self.msg_d.height = IMAGE_HEIGHT
        self.msg_d.width = IMAGE_WIDTH
        self.msg_d.data = self.bridge_d.cv2_to_imgmsg(img_depth, "32FC1").data
        self.msg_d.is_bigendian = 0
        self.msg_d.step = self.msg_d.width * 4
        return self.msg_d

    def CreateInfoMessage(self):
        self.msg_info.header.frame_id = "camera_rgb_optical_frame"
        self.msg_info.height = self.msg_rgb.height
        self.msg_info.width = self.msg_rgb.width
        self.msg_info.distortion_model = "plumb_bob"

        self.msg_info.D.append(CAMERA_K1)
        self.msg_info.D.append(CAMERA_K2)
        self.msg_info.D.append(CAMERA_P1)
        self.msg_info.D.append(CAMERA_P2)
        self.msg_info.D.append(CAMERA_P3)

        self.msg_info.K[0] = CAMERA_FX
        self.msg_info.K[1] = 0
        self.msg_info.K[2] = CAMERA_CX
        self.msg_info.K[3] = 0
        self.msg_info.K[4] = CAMERA_FY
        self.msg_info.K[5] = CAMERA_CY
        self.msg_info.K[6] = 0
        self.msg_info.K[7] = 0
        self.msg_info.K[8] = 1

        self.msg_info.R[0] = 1
        self.msg_info.R[1] = 0
        self.msg_info.R[2] = 0
        self.msg_info.R[3] = 0
        self.msg_info.R[4] = 1
        self.msg_info.R[5] = 0
        self.msg_info.R[6] = 0
        self.msg_info.R[7] = 0
        self.msg_info.R[8] = 1

        self.msg_info.P[0] = CAMERA_FX
        self.msg_info.P[1] = 0
        self.msg_info.P[2] = CAMERA_CX
        self.msg_info.P[3] = 0
        self.msg_info.P[4] = 0
        self.msg_info.P[5] = CAMERA_FY
        self.msg_info.P[6] = CAMERA_CY
        self.msg_info.P[7] = 0
        self.msg_info.P[8] = 0
        self.msg_info.P[9] = 0
        self.msg_info.P[10] = 1
        self.msg_info.P[11] = 0

        self.msg_info.binning_x = self.msg_info.binning_y = 0
        self.msg_info.roi.x_offset = self.msg_info.roi.y_offset = self.msg_info.roi.height = self.msg_info.roi.width = 0
        self.msg_info.roi.do_rectify = False
        self.msg_info.header.stamp = self.msg_rgb.header.stamp
        return self.msg_info

    def CreateTFMessage(self):
        self.msg_tf.transforms.append(TransformStamped())
        self.msg_tf.transforms[0].header.stamp = self.ros_time
        self.msg_tf.transforms[0].header.frame_id = "/camera_link"
        self.msg_tf.transforms[0].child_frame_id = "/camera_rgb_frame"
        self.msg_tf.transforms[0].transform.translation.x = 0.000
        self.msg_tf.transforms[0].transform.translation.y = 0
        self.msg_tf.transforms[0].transform.translation.z = 0.000
        self.msg_tf.transforms[0].transform.rotation.x = 0.00
        self.msg_tf.transforms[0].transform.rotation.y = 0.00
        self.msg_tf.transforms[0].transform.rotation.z = 0.00
        self.msg_tf.transforms[0].transform.rotation.w = 1.00

        self.msg_tf.transforms.append(TransformStamped())
        self.msg_tf.transforms[1].header.stamp = self.ros_time
        self.msg_tf.transforms[1].header.frame_id = "/camera_rgb_frame"
        self.msg_tf.transforms[1].child_frame_id = "/camera_rgb_optical_frame"
        self.msg_tf.transforms[1].transform.translation.x = 0.000
        self.msg_tf.transforms[1].transform.translation.y = 0.000
        self.msg_tf.transforms[1].transform.translation.z = 0.000
        self.msg_tf.transforms[1].transform.rotation.x = -0.500
        self.msg_tf.transforms[1].transform.rotation.y = 0.500
        self.msg_tf.transforms[1].transform.rotation.z = -0.500
        self.msg_tf.transforms[1].transform.rotation.w = 0.500

        self.msg_tf.transforms.append(TransformStamped())
        self.msg_tf.transforms[2].header.stamp = self.ros_time
        self.msg_tf.transforms[2].header.frame_id = "/camera_link"
        self.msg_tf.transforms[2].child_frame_id = "/camera_depth_frame"
        self.msg_tf.transforms[2].transform.translation.x = 0
        self.msg_tf.transforms[2].transform.translation.y = 0
        self.msg_tf.transforms[2].transform.translation.z = 0
        self.msg_tf.transforms[2].transform.rotation.x = 0.00
        self.msg_tf.transforms[2].transform.rotation.y = 0.00
        self.msg_tf.transforms[2].transform.rotation.z = 0.00
        self.msg_tf.transforms[2].transform.rotation.w = 1.00

        self.msg_tf.transforms.append(TransformStamped())
        self.msg_tf.transforms[3].header.stamp = self.ros_time
        self.msg_tf.transforms[3].header.frame_id = "/camera_depth_frame"
        self.msg_tf.transforms[3].child_frame_id = "/camera_depth_optical_frame"
        self.msg_tf.transforms[3].transform.translation.x = 0.000
        self.msg_tf.transforms[3].transform.translation.y = 0.000
        self.msg_tf.transforms[3].transform.translation.z = 0.000
        self.msg_tf.transforms[3].transform.rotation.x = -0.500
        self.msg_tf.transforms[3].transform.rotation.y = 0.500
        self.msg_tf.transforms[3].transform.rotation.z = -0.500
        self.msg_tf.transforms[3].transform.rotation.w = 0.500
        return self.msg_tf


if __name__ == "__main__":
    basic_height_uav = -4977

    f = open("basic_url.txt", "r")
    basic_url = f.read()
    f.close()

    ws = websocket.WebSocket()
    ws.connect("ws://"+ basic_url +"/UAVHeightWebSocket2/1/1")

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    rospy.init_node('airsim1_publisher', anonymous=True)
    # publisher_d1 = rospy.Publisher('/airsim/camera_1/depth_registered/image_raw', Image, queue_size=1)
    publisher_rgb1 = rospy.Publisher('/airsim/camera_1/rgb/image_rect_color', Image, queue_size=1)
    publisher_info1 = rospy.Publisher('/airsim/camera_1/rgb/camera_info', CameraInfo, queue_size=1)
    publisher_tf1 = rospy.Publisher('/airsim/camera_1/tf', TFMessage, queue_size=1)
    #publisher_d2 = rospy.Publisher('/airsim/camera_2/depth_registered/image_raw', Image, queue_size=1)
    publisher_rgb2 = rospy.Publisher('/airsim/camera_2/rgb/image_rect_color', Image, queue_size=1)
    publisher_info2 = rospy.Publisher('/airsim/camera_2/rgb/camera_info', CameraInfo, queue_size=1)
    publisher_tf2 = rospy.Publisher('/airsim/camera_2/tf', TFMessage, queue_size=1)
    rate = rospy.Rate(100)  # 30hz
    pub1 = KinectPublisher()
    pub2 = KinectPublisher()

    uav_height_pub = rospy.Publisher('/airsim/uav_height', String, queue_size=10)

    while not rospy.is_shutdown():
        responses = client.simGetImages([# airsim.ImageRequest("camera_1", airsim.ImageType.DepthPlanar, True, False),
                                         airsim.ImageRequest("camera_1", airsim.ImageType.Scene, False, False),
                                         airsim.ImageRequest("camera_2", airsim.ImageType.Scene, False, False)])
        #img_depth1 = pub1.getDepthImage(responses[0])
        #img_rgb1 = pub1.getRGBImage(responses[1])
        
        img_rgb1 = pub1.getRGBImage(responses[0])

        #img_depth2 = pub2.getDepthImage(responses[2])
        img_rgb2 = pub2.getRGBImage(responses[1])

        if CLAHE_ENABLED:
            img_rgb1 = pub1.enhanceRGB(img_rgb1)
            #img_rgb2 = pub2.enhanceRGB(img_rgb2)

        pub1.GetCurrentTime()
        pub2.GetCurrentTime()
        try:
            msg_rgb1 = pub1.CreateRGBMessage(img_rgb1)
            msg_rgb2 = pub2.CreateRGBMessage(img_rgb2)
        except:
            print("divide zero")
        #msg_d1 = pub1.CreateDMessage(img_depth1)
        msg_info1 = pub1.CreateInfoMessage()
        msg_info2 = pub2.CreateInfoMessage()
        msg_tf1 = pub1.CreateTFMessage()
        msg_tf2 = pub2.CreateTFMessage()

        #msg_rgb2 = pub2.CreateRGBMessage(img_rgb2)
        #msg_d2 = pub2.CreateDMessage(img_depth2)
        #msg_info2 = pub2.CreateInfoMessage()
        #msg_tf2 = pub2.CreateTFMessage()

        publisher_rgb1.publish(msg_rgb1)
        publisher_rgb2.publish(msg_rgb2)
        #publisher_d1.publish(msg_d1)
        publisher_info1.publish(msg_info1)
        publisher_info2.publish(msg_info2)
        publisher_tf1.publish(msg_tf1)
        publisher_tf2.publish(msg_tf2)

        #publisher_rgb2.publish(msg_rgb2)
        #publisher_d2.publish(msg_d2)
        #publisher_info2.publish(msg_info2)
        #publisher_tf2.publish(msg_tf2)
        
        
        state = client.getMultirotorState()
        actual_height = int(int(-state.kinematics_estimated.position.z_val) * 100 - basic_height_uav)
        height = round(actual_height, -1)
        print(actual_height, height)
        ws.send(str(height))
        uav_height_pub.publish(str(height))
        

        del pub1.msg_info.D[:]
        del pub1.msg_tf.transforms[:]

        del pub2.msg_info.D[:]
        del pub2.msg_tf.transforms[:]

        rate.sleep()

    ws.close()





