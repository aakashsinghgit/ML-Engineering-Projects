from flask import Flask, render_template, url_for, redirect, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from predict.predict_views import predict_view

app = Flask(__name__, template_folder='templates')
model = pickle.load(open('model.pkl', 'rb')) # loading the trained model

ENV = 'dev'
app.debug = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:yourpass@localhost:3306/login_store'
app.config['SECRET_KEY'] = 'giveanysceretkey'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

## for managing our application and Flask_login to work together
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

## To reload the objects from the user_ids stored in the session
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

## Create tables in the database
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


##Creating table for storing user details and predicted outcome

class User_details(db.Model, UserMixin):
    user_id = db.Column(db.String(20), primary_key=True)
    gender = db.Column(db.String(20), nullable=False )
    married = db.Column(db.String(20), nullable=False )
    dependents = db.Column(db.Integer , nullable=False )
    education = db.Column(db.String(20), nullable=False )
    self_employed = db.Column(db.String(20), nullable=False )
    Applicant_Income = db.Column(db.Integer , nullable=False )
    Coapplicant_Income = db.Column(db.Integer , nullable=False )
    Loan_amount = db.Column(db.Integer , nullable=False )
    Loan_Amount_Term = db.Column(db.Integer , nullable=False )
    Credit_History = db.Column(db.String(20), nullable=False )
    Property_Area  = db.Column(db.String(20), nullable=False )
    loan_approved = db.Column(db.String(20), nullable=False )

with app.app_context():
    db.create_all()

## Create a RegisterForm
class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError('That username already exists. Please choose a different one.')


## Create a loginform
class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/in_db')
def in_db():
    my_var = session.get('my_var', None)
    int_features = my_var[0]
    my_predict = my_var[1]
    
    new = User_details(user_id =  user_name, 
                        gender = str(int_features[0]), 
                        married = str(int_features[1]), 
                        dependents = int_features[2], 
                        education = str(int_features[3]), 
                        self_employed = str(int_features[4]), 
                        Applicant_Income = int_features[5], 
                        Coapplicant_Income = int_features[6], 
                        Loan_amount = int_features[7], 
                        Loan_Amount_Term = int_features[8], 
                        Credit_History = str(int_features[9]), 
                        Property_Area =str(int_features[10]), 
                        loan_approved = my_predict)
    db.session.add(new)
    db.session.commit()
    return render_template('predict.html', stored_text='Registered in Db')

user_name = "error"
@app.route('/login', methods=['GET', 'POST'])
def login():
    global user_name
    form = LoginForm()
    user_name = form.username.data
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('prediction.enter_details'))
    return render_template('login.html', form=form)


## hashing the password is optional for the learners
@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data) ## creates a password hash
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/get_user_count')
def get_user_count():
    return redirect(url_for('user_count'))

# CREATE THE METADATA OBJECT TO ACCESS THE TABLE
@app.route('/user_count')
def user_count():
    result = db.session.query(User).count()
    return jsonify({'Number of registered users': result})

#Creating stats summary end_point

@app.route('/stat_summary')
def stat_summary():
    result = User_details.query.all()
    return render_template('stat_summary.html', stat_details = result )

#Creating stats summary report end_point
@app.route('/summary_report')
def summary_report():
    count_tot  = db.session.query(User_details).count()
    count_y= User_details.query.filter_by(loan_approved ="Y").count()
    per = (count_y/count_tot)*100

    return render_template('summary_report.html', count = str(count_tot), perc = str(per))

# register the blueprints
app.register_blueprint(predict_view)

if __name__ == "__main__":
    app.run(debug=True,port=6622)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
