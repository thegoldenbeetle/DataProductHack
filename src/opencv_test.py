
import os
import cv2 as cv


def check_source(cap):
    if not cap.isOpened():
        raise Exception('Error opening video stream or file')


def display_video(cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv.imshow('Frame', frame)

            # TODO: Модель и отправка результатов
            # model_output = model(frame)
            # send_output(model_output)

            if cv.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv.destroyAllWindows()


def test_video(source):
    v_cap = cv.VideoCapture(source)
    check_source(v_cap)
    display_video(v_cap)


if __name__ == '__main__':
    DATA_FOLDER = '../data/2_self_captured_data/itmo_cfr_720'
    FILENAME = 'itmo_3_cfr_720.mp4'
    FILEPATH = os.path.join(DATA_FOLDER, FILENAME)

    STREAM_ADDRESS = '127.0.0.1:8554'
    RTSP_ADDRESS = f'rtsp://{STREAM_ADDRESS}/stream'

    # test_video(FILEPATH)
    test_video(RTSP_ADDRESS)
