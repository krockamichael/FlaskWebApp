from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from __init__ import db, picam2
from models import User


auth = Blueprint('auth', __name__)


@auth.route('/login')
def login():
    return render_template('login.html')


@auth.route('/login', methods=['POST'])
def login_post():
    email = request.form.get('email')
    password = request.form.get('password')
    remember = True if request.form.get('remember') else False
    user = User.query.filter_by(email=email).first()

    if not user or not check_password_hash(user.password, password):
        flash('Please check your login details and try again.', 'error')
        return redirect(url_for('auth.login'))

    login_user(user, remember=remember)
    return redirect(url_for('main.index'))


@auth.route('/signup')
def signup():
    return render_template('signup.html')


@auth.route('/signup', methods=['POST'])
def signup_post():
    email = request.form.get('email')
    name = request.form.get('name')
    password = request.form.get('password')
    user = User.query.filter_by(email=email).first()

    if user:
        flash('Email address already exists')
        return redirect(url_for('auth.signup'))

    new_user = User(email=email, name=name, password=generate_password_hash(password, method='sha256'))
    db.session.add(new_user)
    db.session.commit()

    return redirect(url_for('auth.login'))


@auth.route('/logout')
@login_required
def logout():
    if picam2 is not None:
        picam2.stop_recording()
    logout_user()
    return redirect(url_for('auth.login'))


@auth.route('/profile')
@login_required
def profile():
    if picam2 is not None:
        picam2.stop_recording()
    return render_template('profile.html', name=current_user.name)


@auth.route('/profile', methods=['PUT', 'POST'])
@login_required
def profile_put():
    email = request.form.get('email')
    name = request.form.get('name')
    new_password = request.form.get('new_password')
    old_password = request.form.get('old_password')
    user = User.query.filter_by(email=email).first()

    if user and check_password_hash(user.password, old_password):
        flash_msg = ''
        if email and email != user.email:
            setattr(user, 'email', email)
            flash_msg += 'Email' if flash_msg == '' else ', Email'
        if name and name != user.name:
            setattr(user, 'name', name)
            flash_msg += 'Name' if flash_msg == '' else ', Name'
        if new_password and new_password != old_password:
            setattr(user, 'password', generate_password_hash(new_password, method='sha256'))
            flash_msg += 'Password' if flash_msg == '' else ', Password'

        if flash_msg == '':
            flash('No changes detected', 'error')
            return render_template('profile.html', name=current_user.name)

        db.session.commit()
        flash(f'User data: {flash_msg} has been changed!', 'ok')
        return redirect(url_for('auth.profile'))

    if not user:
        flash(f'User not found.', 'error')
    if not check_password_hash(user.password, old_password):
        flash('Please enter your password.', 'error') if old_password == '' else flash('Incorrect password.', 'error')

    return render_template('profile.html', name=current_user.name)
