from flask import Flask, render_template
# from flask_login import LoginManager
import keras
import numpy as np
# from picamera import PiCamera

# camera = PiCamera()

MODEL_PATH = "models\keras_mAlexNet\out_keras.h5"

app = Flask(__name__)
# model = keras.models.load_model(MODEL_PATH)

# lm = LoginManager()
# lm.init_app(app)


@app.route("/")
def index():
    return render_template("index.html")

def predict(image) -> str:
    try:
        return "occupied" if np.argmax(model.predict(image)) == 1 else "empty"
    except ValueError:
        return "Image is invalid!"

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)