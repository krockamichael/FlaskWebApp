from core_logic import check_occupancy, DrawRectangles
from __init__ import picam2_resolution
from picamera2.picamera2 import *
from threading import Condition
import time
import io


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


def init_camera():
    picam2 = Picamera2()
    picam2.start_preview()
    config = picam2.video_configuration(main={"size": picam2_resolution})
    picam2.request_callback = DrawRectangles
    picam2.configure(config)
    return picam2


def gather_img(picam2, output):
    while True:
        with output.condition:
            # time.sleep(0.1)
            output.condition.wait()
            check_occupancy(picam2, time.time())
            frame = output.frame
        yield (b'--frame\r\nContent-Type:image/jpeg\r\n\r\n' + frame + b'\r\n')
