#!/bin/bash

FPS=30
VCODEC="libx264"                # libx264, libx265, mpeg4, libxvid
STREAM_ADDRESS="127.0.0.1:8554"
filepath="./test_1.mp4"

ffmpeg -re -i $filepath -filter:v fps=$FPS -vcodec $VCODEC -f rtsp -rtsp_transport tcp "rtsp://$STREAM_ADDRESS/stream"
