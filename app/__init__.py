from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from core_logic import init_model
from flask_mail import Mail
from flask import Flask


# setup db, mail, dummy camera
db = SQLAlchemy()
mail = Mail()
picam2 = None
picam2_resolution = (2240, 1680)# (1280, 960) # (640, 480)

# load tflite model. initialize segmentation mask overlay
init_model('/home/michael/Desktop/VNOS/model/model.tflite',
           '/home/michael/Desktop/VNOS/data/segmentation_mask.csv',
           picam2_resolution)


def create_app():
    app = Flask(__name__)

    app.config['SECRET_KEY'] = 'secret-key-goes-here'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    app.config['MAIL_SERVER'] = 'smtp.gmail.com'
    app.config['MAIL_PORT'] = 465
    app.config['MAIL_USERNAME'] = 'krocka.michael@gmail.com'
    app.config['MAIL_PASSWORD'] = 'dummy_password'
    app.config['MAIL_USE_TLS'] = False
    app.config['MAIL_USE_SSL'] = True

    db.init_app(app)
    mail.init_app(app)

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    from models import User
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    from auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint)

    from main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
