#!/bin/bash
echo "[bash] rosrun image_transport republish raw in:=/airsim/camera_1/rgb/image_rect_color compressed out:=/airsim/camera_1/compressed/rgb/image_rect_color"
rosrun image_transport republish raw in:=/airsim/camera_1/rgb/image_rect_color compressed out:=/airsim/camera_1/compressed/rgb/image_rect_color
