#!/usr/bin/env python3.8
# #指定Python运行，即Python3
# coding:utf‐8 #设置编码格式为utf‐8
# python uav control
import sys
import time
import airsim
import pygame
import rospy
import websocket

from std_msgs.msg import Int32

global AirSim_client
AirSim_client = None

def callback(data):
    # print the actual message in its raw format
    rospy.loginfo("xy_frame_num: %d", data.data)

    # 设置速度
    scale_ratio = 1.0

    # 基础的偏航速率
    base_yaw_rate = 5.0

    yaw_rate = 0.0
    velocity_x = 0.0
    velocity_y = 0.0
    velocity_z = 0.0

    num = data.data

    # for event in pygame.event.get():
        # if event.type == pygame.QUIT:
            # sys.exit()

    # 接收到-1表示俯视摄像头已经完成火焰食品录制，当前飞控任务结束
    if num == -1:
        # pygame.quit()
        rospy.signal_shutdown("camera1_uav_control finished!")
        sys.exit()

    # 上升
    if num == 100:
        velocity_z = -1 * scale_ratio

    # 下降
    if num == -100:
        velocity_z = 1 * scale_ratio

    # 根据 'Q' 和 'E' 按键来设置偏航速率变量
    # if scan_wrapper[pygame.K_q] or scan_wrapper[pygame.K_e]:
    if num == 1 or num == 4 or num == 7:
        velocity_y = -1 * scale_ratio

    if num == 3 or num == 6 or num == 9:
        velocity_y = 1 * scale_ratio

    # 根据 'W' 和 'S' 按键来设置pitch轴速度变量(NED坐标系，x为机头向前)
    # if scan_wrapper[pygame.K_w] or scan_wrapper[pygame.K_s]:
    #     velocity_x = (scan_wrapper[pygame.K_w] - scan_wrapper[pygame.K_s]) * scale_ratio

    # 根据 'A' 和 'D' 按键来设置roll轴速度变量(NED坐标系，y为正右方)
    # if scan_wrapper[pygame.K_a] or scan_wrapper[pygame.K_d]:
    #     velocity_y = -(scan_wrapper[pygame.K_a] - scan_wrapper[pygame.K_d]) * scale_ratio

    # 根据 'SPACE' 和 'LCTRL' 按键来设置z轴速度变量(NED坐标系，z轴向上为负)
    # if scan_wrapper[pygame.K_SPACE] or scan_wrapper[pygame.K_LCTRL]:
    if num == 1 or num == 2 or num == 3:
        velocity_x = 1 * scale_ratio

    if num == 7 or num == 8 or num == 9:
        velocity_x = -1 * scale_ratio
    # print(f": Expectation gesture: {velocity_x}, {velocity_y}, {velocity_z}, {yaw_rate}")

    # 设置速度控制以及设置偏航控制(存在一定问题，大家测试过才知道)
    global AirSim_client
    AirSim_client.moveByVelocityBodyFrameAsync(vx=velocity_x, vy=velocity_y, vz=velocity_z, duration=0.02,
                                               yaw_mode=airsim.YawMode(True, yaw_or_rate=yaw_rate),
                                               vehicle_name=vehicle_name)

    ws.send("无人机火焰捕获")


def main():
    rospy.init_node('camera2_uav_control', anonymous=True)
    rospy.Subscriber("/airsim/camera_2/xy_frame", Int32, callback)

    rospy.spin()


if __name__ == '__main__':
    # >------>>>  pygame settings   <<<------< #
    # pygame.init()
    # screen = pygame.display.set_mode((640, 480))
    # pygame.display.set_caption('keyboard ctrl')
    # screen.fill((0, 0, 0))

    # >------>>>  AirSim settings   <<<------< #
    # 这里改为你要控制的无人机名称(settings文件里面设置的)
    vehicle_name = "Drone_1"

    AirSim_client = airsim.MultirotorClient()
    print(AirSim_client.listVehicles())
    AirSim_client.confirmConnection()
    AirSim_client.enableApiControl(True, vehicle_name=vehicle_name)
    AirSim_client.armDisarm(True, vehicle_name=vehicle_name)
    AirSim_client.takeoffAsync(vehicle_name=vehicle_name).join()

    f = open("basic_url.txt", "r")
    basic_url = f.read()
    f.close()

    ws = websocket.WebSocket()
    ws.connect("ws://"+ basic_url +"/UAVHeightWebSocket/1/1")
    
    

    try:
        main()
        ws.close()
    except rospy.ROSInterruptException:
        pass
