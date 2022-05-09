import tflite_runtime.interpreter as tflite
# from ALPR import PlateDetector, PlateRecognizer
from typing import Union
import pandas as pd
import numpy as np
import datetime
import time
import cv2


# IMPORTANT #
# ALPR is commented out because the pytesseract_to_string method takes too long
# The implementation using PyTesseract is not suitable for devices with low computational power

interpreter = None
input_details = None
output_details = None
height, width = 224, 224
floating_model = True

time_stamp = time.time()
segmented_images = []
statuses = []
points = []
colors = []

# plate_detector = None
# plate_recognizer = None
# lp_dict = {}


def init_model(model_path:str, seg_mask_path:str, picam2_resolution:tuple) -> None:
    global interpreter, input_details, output_details, width, height, floating_model, statuses, colors
    # global plate_detector, plate_recognizer

    # initialize points, statuses and box colors
    init_points(pd.read_csv(seg_mask_path), picam2_resolution)
    statuses = ['Empty'] * len(points)
    colors = list(((0,255,0,0), ) * len(points))

    # initialize model
    interpreter = tflite.Interpreter(model_path=model_path, num_threads=4)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    interpreter.resize_tensor_input(0, [len(points), width, height, 3])
    interpreter.allocate_tensors()
    floating_model = False
    if input_details[0]['dtype'] == np.float32:
        floating_model = True

    # plate_detector = PlateDetector()
    # plate_recognizer = PlateRecognizer()


def init_points(seg_mask, picam2_resolution:tuple) -> None:
    def res_conv(value:int) -> int:
        # scale to current resolution
        original_image_resolution = (1920, 1440)
        ratio = picam2_resolution[0] / original_image_resolution[0]
        return int(value * ratio)

    for _, row in seg_mask.iterrows():
        left_top_point = (res_conv(row['left_top_x']), res_conv(row['left_top_y']))
        right_bottom_point = (res_conv(row['right_bottom_x']), res_conv(row['right_bottom_y']))
        points.append([left_top_point, right_bottom_point])


def preprocess_image(picam2) -> list:
    global segmented_images
    # get RGB image from picamera
    bgr = picam2.capture_array()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # apply segmentation mask and preprocess images
    input_data_list = []
    segmented_images = segment_parking_lot(rgb)
    for parking_space in segmented_images:
        picture = cv2.resize(parking_space, (height, width))
        input_data = np.expand_dims(picture, axis=0)
        if floating_model:
            input_data = np.float32(input_data / 255)
        input_data_list.append(input_data)

    del rgb, bgr
    return input_data_list


def segment_parking_lot(image:np.ndarray) -> list:
    segmented_images = []

    for point in points:
        seg_image = image[point[0][1]:point[1][1], point[0][0]:point[1][0]]
        segmented_images.append(seg_image)

    return segmented_images


def get_model_prediction(input_data_list:list) -> Union[str, list]:
    global points, width, height
    input_data = np.array(input_data_list).reshape((len(points), width, height, 3))
    interpreter.set_tensor(0, input_data)
    interpreter.invoke()
    new_statuses, confidence_list = [], []

    for i in interpreter.get_tensor(output_details[0]['index']):
        new_statuses.append("Occupied" if np.argmax(i) == 1 else "Empty")
        confidence_list.append([round(i[0], 3), round(i[1], 3)])

    return new_statuses, confidence_list


def execute_occupancy_logic(picam2) -> list:
    global statuses, colors #, lp_dict
    input_data_list = preprocess_image(picam2)
    new_statuses, confidence_list = get_model_prediction(input_data_list)
    messages = []
    
    for i, value in enumerate(zip(new_statuses, statuses, confidence_list)):
        new_status, status, confidence = value[0], value[1], value[2]
        colors[i] = (255,0,0,0) if new_status == 'Occupied' else (0,255,0,0)

        if new_status != status:
            msg = f'{str(datetime.datetime.now())} INFO Status of parking space {i} changed from {status} to {new_status} with confidence {confidence}'
            print(msg)
            messages.append(msg)

            # ALPR
            # if new_status == 'Occupied' and segmented_images[i] is not None:
            #    license_plates = plate_detector.get_license_plate_candidates(segmented_images[i])
            #    detected_text = plate_recognizer.get_text(license_plates)
            #    print(detected_text)
            #    if detected_text is not None:
            #        lp_dict[i] = detected_text

    if not messages:
        messages = f'{datetime.datetime.now()} INFO No parking status change'
        print(messages)
    else: # update statuses, use new_s if it was changed, else keep old_s
        statuses = [new_s if new_s != '' else old_s for new_s, old_s in zip(new_statuses, statuses)]

    return messages


def check_occupancy(picam2, new_time_stamp=None, period:int=10) -> list:
    global time_stamp
    if picam2 is not None:
        if new_time_stamp is not None:
            if int(new_time_stamp - time_stamp) > period:
                time_stamp = new_time_stamp
                return execute_occupancy_logic(picam2)
        else:
            return execute_occupancy_logic(picam2)


def DrawRectangles(request):
    global points, colors   #, lp_dict
    stream = request.picam2.stream_map['main']
    fb = request.request.buffers[stream]
    size = tuple(reversed(request.picam2.camera_config['main']['size'])) + (4,)

    with fb.mmap(0) as b:
        im = np.array(b, copy=False, dtype=np.uint8).reshape(size) # (960, 1280) (1440, 1920)

        for i, value in enumerate(zip(colors, points)):
            color, point = value[0], value[1]
            cv2.rectangle(im, point[0], point[1], color)

            # coords are top center of bounding box for parking space
            coords = (int(point[0][0] + (point[1][0] - point[0][0]) / 2), point[0][1])
            cv2.putText(im, str(i), coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            # if i in lp_dict:
            #    coords = (point[0][0], point[0][1] + (point[0][1] - point[1][1]))
            #    cv2.putText(im, lp_dict[i], coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        del im