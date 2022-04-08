# from picamera import PiCamera
# camera = PiCamera()
from flask import Blueprint, render_template, redirect, url_for, flash
from flask_login import login_required, current_user
from flask_mail import Message
from keras.preprocessing import image
import numpy as np
from __init__ import create_app, keras_model, mail


main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html', name=current_user.name)

@main.route('/predict')
def predict() -> str:
    path = "test_data\empty.jpg" # empty.jpg
    img = image.load_img(path, target_size=(224,224))
    image_array = image.img_to_array(img)
    image_array = image_array * 1. / 256 # convert to float and normalize
    image_batch = np.expand_dims(image_array, axis=0)
    try:
        status = "occupied" if np.argmax(keras_model.predict(image_batch)) == 1 else "empty"
        flash(status)

        msg = Message('Hello from flask parking app',
                body=status,
                sender=current_user.email,
                recipients=['dominikasvedova@gmail.com', current_user.email])
        mail.send(msg)

        return redirect(url_for('main.index'))
    except ValueError:
        flash("ValueError")
        return redirect(url_for('main.index'))

if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=8080, debug=True)