#!/bin/bash
gnome-terminal --geometry 60x20+10+10 -- bash yjn_roscore.sh & sleep 3
gnome-terminal --geometry 60x20+10+10 -- bash yjn_publish1.sh & sleep 2
gnome-terminal --geometry 60x20+10+10 -- bash yjn_img_convert1.sh & sleep 2
gnome-terminal --geometry 60x20+10+10 -- bash yjn_img_convert2.sh & sleep 2
#gnome-terminal --geometry 60x20+10+10 -- bash yjn_keyboard.sh & sleep 2
gnome-terminal --geometry 60x20+10+10 -- bash yjn_front_camera.sh & sleep 2
gnome-terminal --geometry 60x20+10+10 -- bash yjn_below_camera.sh & sleep 2
gnome-terminal --geometry 60x20+10+10 -- bash yjn_uav_ctrl1.sh & sleep 2
gnome-terminal --geometry 60x20+10+10 -- bash yjn_uav_ctrl2.sh & sleep 2

