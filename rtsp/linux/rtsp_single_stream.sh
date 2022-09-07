#!/bin/bash

FPS=30
VCODEC="libx264"                # libx264, libx265, mpeg4, libxvid
STREAM_ADDRESS="127.0.0.1:8554"
filepath="../../data/2_self_captured_data/itmo_cfr_720/itmo_3_cfr_720.mp4"

ffmpeg -re -i $filepath -filter:v fps=$FPS -vcodec $VCODEC -f rtsp -rtsp_transport tcp "rtsp://$STREAM_ADDRESS/stream"
