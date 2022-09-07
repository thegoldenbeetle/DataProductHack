import time
import os
import cv2
import json
import datetime

from paddleocr import PaddleOCR

import sys
sys.path.insert(0, './darknet')
import darknet
import darknet_video



def check_source(cap):
    if not cap.isOpened():
        raise Exception('Error opening video stream or file')


def display_video(cap):

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            #cv.imshow('Frame', frame)

            # TODO: Модель и отправка результатов
            # model_output = model(frame)
            # send_output(model_output)

            # image preprocess for yolo
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                    interpolation=cv2.INTER_LINEAR)
            img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
            darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())

            # detect license plates
            detections = darknet.detect_image(network, class_names, img_for_detect, thresh=thresh)

            # postprocess of yolo output
            detections_adjusted = []
            for label, confidence, bbox in detections:
                bbox_adjusted = darknet_video.convert2original_with_params(frame, bbox, darknet_height, darknet_width)
                detections_adjusted.append((str(label), confidence, bbox_adjusted))

            image = frame.copy()
            
            # ocr
            for detection in detections_adjusted:
                bbox = detection[2]
                left, top, right, bottom = darknet.bbox2points(bbox)
                cr_img = frame[top:bottom, left:right]
                
                car_number_str = ''
                try:
                    result = ocr.ocr(cr_img, cls=False, det=False)
                    car_number_str = result[0][0]
                    print('result', result)
                except ZeroDivisionError:
                    print('Ocr error!')

                if car_number_str:
                    # sent json
                    timestamp = datetime.datetime.now()
                    dict_to_server = {'timestamp': str(timestamp), 'number': car_number_str}
                    json_to_server = json.dumps(dict_to_server)

                    # get and draw image with license box
                    (label_width, label_height), baseline = cv2.getTextSize(car_number_str, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
                    top_left = tuple(map(int,[int(bbox[0]),int(bbox[1])-(label_height+baseline)]))
                    top_right = tuple(map(int,[int(bbox[0])+label_width,int(bbox[1])]))
                    org = tuple(map(int,[int(bbox[0]),int(bbox[1])-baseline]))
                    cv2.putText(image, car_number_str, org, cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)

            image = darknet.draw_boxes(detections_adjusted, image, class_colors)
            cv2.imshow('Frame', image)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def test_video(source):
    v_cap = cv2.VideoCapture(source)
    check_source(v_cap)
    display_video(v_cap)


if __name__ == '__main__':
    DATA_FOLDER = '../data/2_self_captured_data/itmo_cfr_720'
    FILENAME = 'itmo_3_cfr_720.mp4'
    FILEPATH = os.path.join(DATA_FOLDER, FILENAME)

    STREAM_ADDRESS = '127.0.0.1:8554'
    RTSP_ADDRESS = f'rtsp://{STREAM_ADDRESS}/stream'

    # for detecting license plates
    config_file = './darknet/cfg/yolov4-obj.cfg'
    data_file = './darknet/data/obj.data'
    weights = './darknet/checkpoint/yolov4-obj_best.weights'
    thresh = 0.25
    network, class_names, class_colors = darknet.load_network(
            config_file,
            data_file,
            weights,
            batch_size=1
        )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)

    # for number recognition
    ocr = PaddleOCR(lang='en',rec_algorithm='CRNN')

    # test_video(FILEPATH)
    test_video(RTSP_ADDRESS)
