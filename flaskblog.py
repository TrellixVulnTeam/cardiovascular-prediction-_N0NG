import os

#import tensorflow as tf
from flask import Flask, render_template, url_for, request, flash, redirect
from forms import RegistrationForm, LoginForm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# from keras.utils import np_utils
import os
import pickle
from flask_wtf import FlaskForm
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from datetime import date
# from _thread import *
from keras.models import model_from_json
from keras import backend as K
app = Flask(__name__)


posts = [

    {
        'author': 'kanishka sahu',
        'title': 'post titile',
        'content': 'post content',
        'date': '21 Sep 2019'
    },
    {
        'author': 'kanishka sahu',
        'title': 'post titile',
        'content': 'post content',
        'date': '21 Sep 2019'
    },
    {
        'author': 'kanishka sahu',
        'title': 'post titile',
        'content': 'post content',
        'date': ' 21 Sep 2019'
    },
    {
        'author': 'kanishka sahu',
        'title': 'post titile',
        'content': 'post content',
        'date': '21 Sep 2019'
    }
]
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


@app.route('/')
@app.route('/home')
def home():
    return render_template('layout.html', posts=posts)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/result', methods=['GET', 'POST'])
def result():
    return render_template('result.html')


@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if form.email.data == 'admin@blog.com' and form.password.data == 'password':
            flash('You have been logged in!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', title='Login', form=form)


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        flash('Account created for {form.username.data}!', 'success')
        return redirect(url_for('home'))
    return render_template('register.html', title='Register', form=form)


@app.route('/p_form', methods=['GET', 'POST'])
def p_form():

    return render_template('newform.html')


class ReusableForm(Form):

    day = TextField('Day:', validators=[validators.required()])
    month = TextField('Month:', validators=[validators.required()])
    year = TextField('Year:', validators=[validators.required()])
    gender = TextField('Gender:', validators=[validators.required()])
    height = TextField('Height:', validators=[validators.required()])
    weight = TextField('weight:', validators=[validators.required()])
    systolicbloodpressure = TextField(
        'systolicbloodpressure:', validators=[validators.required()])
    diastolicbloodpressure = TextField(
        'diastolicbloodpressure:', validators=[validators.required()])
    cholesterol = TextField('cholesterol:', validators=[validators.required()])
    gluc = TextField('gluc', validators=[validators.required()])
    smoke = TextField('Smoke:', validators=[validators.required()])
    alco = TextField('Alco:', validators=[validators.required()])
    active = TextField('Active:', validators=[validators.required()])

    @app.route("/newForm", methods=['GET', 'POST'])
    def hello():
        form = ReusableForm(request.form)

        print(form.errors)
        if request.method == 'POST':

            day = int(request.form['day'])  # day of bith
            month = int(request.form['month'])  # month of bith (in numbers)
            year = int(request.form['year'])  # year of bith
            gender = int(request.form['gender'])  # 0 for women, 1 for men
            height = int(request.form['height'])  # in cm
            weight = int(request.form['weight'])  # in kilograms
            # Systolic blood pressure
            systolicbloodpressure = int(request.form['systolicbloodpressure'])
            # Diastolic blood pressure
            diastolicbloodpressure = int(
                request.form['diastolicbloodpressure'])
            # 1: normal, 2: above normal, 3: well above normal
            cholesterol = int(request.form['cholesterol'])
            # 1: normal, 2: above normal, 3: well above normal
            gluc = int(request.form['gluc'])
            smoke = int(request.form['smoke'])  # 1 if you smoke, 0 if not
            # 1 if you drink alcohol, 0 if not
            alco = int(request.form['alco'])
            # 1 if you do physical activity, 0 if not
            active = int(request.form['active'])

            f_date = date(year, month, day)
            l_date = date.today()
            delta = l_date - f_date
            agedays = delta.days

            agedayscale = -0.23809523809523808
            heightscale = 0.6564102564102564
            weightscale = 0.41578947368421054
            sbpscale = 0.016697588126159554
            dbpscale = 0.013550135501355014
            cholesterolscale = 0.0
            glucscale = 0.0

            single = np.array([agedayscale, gender, heightscale, weightscale,
                               sbpscale, dbpscale, cholesterolscale, glucscale, smoke, alco, active])
            singledf = pd.DataFrame(single)
            final = singledf.transpose()
            # pickle_in = open("data/model.pickle","rb")
            # model = pickle.load(pickle_in)
            # prediction=model.predict(final)
            # print(name,day,month)
            # json_string='{"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "batch_input_shape": [null, 11], "dtype": "float32", "units": 25, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 2, "axis": 0}}, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 5, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 3, "activation": "softsign", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "keras_version": "2.3.1", "backend": "tensorflow"}'
            # modeltopredict = model_from_json(json_string)
            # modeltopredict.load_weights("data/weights.hdf5", by_name=False)
            json_file = open('data/model_config.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("data/weights.hdf5")
            print("Loaded model from disk")
            prediction = loaded_model.predict(final)
            print("*********************************************************************************************", prediction)

            if prediction[0, 0] >= 0.5:
                print("The probability of having or to have a Cardiovascular Disease is: " +
                      str(round(prediction[0, 0]*100, 2)) + "%")
                print("You must visit a doctor to check it :(")
            elif prediction[0, 0] < 0.5 and prediction[0, 0] >= 0.3:
                print("The probability of having or to have a Cardiovascular Disease is: " +
                      str(round(prediction[0, 0]*100, 2)) + "%")
                print("Probably you are healthy :/ ")
            else:
                print("The probability of having or to have a Cardiovascular Disease is: " +
                      str(round(prediction[0, 0]*100, 2)) + "%")
                print("You are healthy :) ")
            K.clear_session()
            # if(prediction==):
            #     text="Normal"
            # elif(prediction==):
            #     text="You have to concious about your health"
            # elif(prediction==):
            #     text="hey concern with doctor youre heart not good"
            # else:
            #     text="Danger boss"

        # if form.validate():
        #     # Save the comment here.
        #     flash('Hello')
        # else:
        #     flash('All the form fields are required. ')

        return render_template('newform.html', form=form)


if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=5000)
