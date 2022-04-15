from flask import Blueprint, render_template, redirect, url_for, flash, Response
from flask_login import current_user, login_required
from flask_mail import Message
from picamera2.picamera2 import *
from picamera2.encoders.jpeg_encoder import *
from threading import Condition
from __init__ import create_app, mail, picam2, interpreter, input_details, output_details, width, height, floating_model
import numpy as np
import cv2
import io


main = Blueprint('main', __name__)

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

def gather_img(output):
    while True:
        with output.condition:
            time.sleep(0.1)
            output.condition.wait()
            frame = output.frame
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@main.route('/')
@login_required
def index():
    if picam2 != '':
        picam2.stop_recording()
    return render_template('index.html')

@main.route('/camera')
@login_required
def camera():
    global picam2
    if picam2 == '':
        picam2 = Picamera2()
        picam2.start_preview()
        config = picam2.video_configuration(main={"size": (640, 480)})
        picam2.configure(config)
    output = StreamingOutput()
    picam2.start_recording(JpegEncoder(), output)
    return Response(gather_img(output), mimetype='multipart/x-mixed-replace; boundary=frame')


@main.route('/predict')
@login_required
def predict():
    try:
        # get image and model prediction
        input_data = preprocess_image()
        status = get_model_prediction(input_data)

        # flash message and send email
        flash(status)
        msg = Message('Hello from flask parking app',
                      body=f"Status changed from \"{'Empty' if status == 'Occupied' else 'Occupied'}\" to \"{status}\".",
                      sender=current_user.email,
                      recipients=[current_user.email])
        mail.send(msg)
        return redirect(url_for('main.index'))

    except ValueError:
        flash("ValueError")
        return redirect(url_for('main.index'))

def preprocess_image():
    # get grey image from picamera
    stride = picam2.stream_configuration("main")["stride"]
    buffer = picam2.capture_buffer("main")
    grey = buffer[:stride * 480].reshape((480, stride))

    # TODO apply mask

    # preprocess image
    rgb = cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)
    picture = cv2.resize(rgb, (width, height))
    input_data = np.expand_dims(picture, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    return input_data

def get_model_prediction(input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    status = "Occupied" if np.argmax(interpreter.get_tensor(output_details[0]['index'])) == 1 else "Empty"
    print("INFO", interpreter.get_tensor(output_details[0]['index']), status)
    return status


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=7000, threaded=True, debug=True)
