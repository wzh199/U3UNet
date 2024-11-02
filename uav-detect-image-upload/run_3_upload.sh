#!/bin/bash
gnome-terminal --geometry 60x20+10+10 -- bash uav_img_front.sh & sleep 2
gnome-terminal --geometry 60x20+10+10 -- bash uav_img_below.sh & sleep 2
gnome-terminal --geometry 60x20+10+10 -- bash uav_img_belowBin.sh & sleep 2
gnome-terminal --geometry 60x20+10+10 -- bash uav_img_below_to_server.sh & sleep 2
