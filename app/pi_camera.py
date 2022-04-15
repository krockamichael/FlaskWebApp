from picamera2.picamera2 import Picamera2
from picamera2.previews.qt_gl_preview import *
import time

picam2 = Picamera2()
picam2.start_preview(QtGlPreview())

preview_config = picam2.preview_configuration()
picam2.configure(preview_config)

picam2.start()
time.sleep(30)
