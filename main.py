from flask import Flask
import keras
import numpy as np

MODEL_PATH = "models\keras_mAlexNet\out_keras.h5"

app = Flask(__name__)
model = keras.models.load_model(MODEL_PATH)

@app.route("/")
def index():
    return "Congratulations, it's a web app!"

def predict(image) -> str:
    return "occupied" if np.argmax(model.predict(image)) == 1 else "empty"

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)