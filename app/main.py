from flask import Blueprint, render_template, redirect, url_for, flash, Response
from __init__ import create_app, mail, picam2, picam2_resolution
from camera import StreamingOutput, init_camera, gather_img
from picamera2.encoders.jpeg_encoder import JpegEncoder
from core_logic import check_occupancy, DrawRectangles
from flask_login import current_user, login_required
from flask_mail import Message


main = Blueprint('main', __name__)


@main.route('/')
@login_required
def index():
    if picam2 is not None:
        picam2.stop_recording()

    return render_template('index.html')


@main.route('/camera')
@login_required
def camera():
    global picam2
    if picam2 is None:
        picam2 = init_camera()
    output = StreamingOutput()
    picam2.start_recording(JpegEncoder(), output)

    return Response(gather_img(picam2, output), mimetype='multipart/x-mixed-replace; boundary=frame')


@main.route('/predict')
@login_required
def predict():
    try:
        # get parking space status changes
        statuses = check_occupancy(picam2)

        # send email
        if statuses is not None:
            msg = Message('Hello from flask parking app',
                          body='\n\n'.join(statuses),
                          sender=current_user.email,
                          recipients=[current_user.email])
            mail.send(msg)
        return redirect(url_for('main.index'))

    except ValueError:
        flash("ValueError")
        return redirect(url_for('main.index'))


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=7000, threaded=True, debug=True)
