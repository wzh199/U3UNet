import sys
import time
import airsim
import pygame

# >------>>>  pygame settings   <<<------< #
pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption('keyboard ctrl')
screen.fill((0, 0, 0))

# >------>>>  AirSim settings   <<<------< #
# 这里改为你要控制的无人机名称(settings文件里面设置的)
vehicle_name = "Drone_1"
AirSim_client = airsim.MultirotorClient()
print(AirSim_client.listVehicles())
AirSim_client.confirmConnection()
AirSim_client.enableApiControl(True, vehicle_name=vehicle_name)
AirSim_client.armDisarm(True, vehicle_name=vehicle_name)
AirSim_client.takeoffAsync(vehicle_name=vehicle_name).join()

# 基础的控制速度(m/s)
base_velocity = 2.0
# 设置临时加速比例
speedup_ratio = 10.0
# 用来设置临时加速
speedup_flag = False

# 基础的偏航速率
base_yaw_rate = 5.0


while True:

    yaw_rate = 0.0
    velocity_x = 0.0
    velocity_y = 0.0
    velocity_z = 0.0

    time.sleep(0.02)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    scan_wrapper = pygame.key.get_pressed()

    # 按下LSHIFT键加速10倍
    if scan_wrapper[pygame.K_LSHIFT]:
        scale_ratio = speedup_ratio
    else:
        scale_ratio = speedup_ratio / speedup_ratio

    # 根据 'Q' 和 'E' 按键来设置偏航速率变量
    if scan_wrapper[pygame.K_q] or scan_wrapper[pygame.K_e]:
        yaw_rate = (scan_wrapper[pygame.K_e] - scan_wrapper[pygame.K_q]) * scale_ratio * base_yaw_rate

    # 根据 'W' 和 'S' 按键来设置pitch轴速度变量(NED坐标系，x为机头向前)
    if scan_wrapper[pygame.K_w] or scan_wrapper[pygame.K_s]:
        velocity_x = (scan_wrapper[pygame.K_w] - scan_wrapper[pygame.K_s]) * scale_ratio

    # 根据 'A' 和 'D' 按键来设置roll轴速度变量(NED坐标系，y为正右方)
    if scan_wrapper[pygame.K_a] or scan_wrapper[pygame.K_d]:
        velocity_y = -(scan_wrapper[pygame.K_a] - scan_wrapper[pygame.K_d]) * scale_ratio

    # 根据 'W' 和 'S' 按键来设置z轴速度变量(NED坐标系，z轴向上为负)
    if scan_wrapper[pygame.K_SPACE] or scan_wrapper[pygame.K_LCTRL]:
        velocity_z = -(scan_wrapper[pygame.K_SPACE] - scan_wrapper[pygame.K_LCTRL]) * scale_ratio

    # print(f": Expectation gesture: {velocity_x}, {velocity_y}, {velocity_z}, {yaw_rate}")

    # 设置速度控制以及设置偏航控制(存在一定问题，大家测试过才知道)
    AirSim_client.moveByVelocityBodyFrameAsync(vx=velocity_x, vy=velocity_y, vz=velocity_z, duration=0.02,
                                      yaw_mode=airsim.YawMode(True, yaw_or_rate=yaw_rate), vehicle_name=vehicle_name)

    # press 'Esc' to quit
    if scan_wrapper[pygame.K_ESCAPE]:
        pygame.quit()
        sys.exit()


