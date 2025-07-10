from flask import Blueprint,render_template,request, session
from flask import render_template
import pickle
import numpy as np



predict_view = Blueprint('prediction', __name__, template_folder="templates")
model = pickle.load(open('model.pkl', 'rb')) # loading the trained model

@predict_view.route('/prediction.enter_details') ## for entering details
def enter_details():
    return render_template('predict.html')
    
# int_features = []#["default","default",0,"default","default",0,0,0,0,"default","default"]

@predict_view.route('/prediction.predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    int_features1 = [float(x) for x in request.form.values()]

    #populating the forms values in a variable to be transferred to db
    list1 = []
    list_temp = ["male", "yes", 0.0, "graduate", "yes",0.0,0.0,0.0,0.0, "yes", "urban" ]
    if int_features1[0] == 0.0:
        list_temp[0] = "female"
    if int_features1[1] == 0.0:
        list_temp[1] = "no"
    list_temp[2] = int_features1[2]
    if int_features1[3] == 0.0:
        list_temp[3] ="not graduate"
    if int_features1[4] == 0.0:
        list_temp[4] = "no"

    list_temp[5] = int_features1[5]
    list_temp[6] = int_features1[6]
    list_temp[7] = int_features1[7]
    list_temp[8] = int_features1[8] 
    
    if int_features1[9] == 0.0:
        list_temp[9] = "no"
    if int_features1[10] == 0.0:
        list_temp[10] = "rural"
    elif int_features1[10] == 2.0:
        list_temp[10] = "semiurban"

    list1.append(list_temp)

    #prediction part
    final_features = [np.array(int_features1)]
    prediction = model.predict(final_features)

    if prediction==0:
        #populating the forms values in a variable to be transferred to db
        predicted = "N"
        list1.append(predicted)
        session['my_var']  = list1

        return render_template('predict.html', 
        prediction_text='Sorry:( you are not eligible for the loan ')
    else:
        #populating the forms values in a variable to be transferred to db
        predicted = "Y"
        list1.append(predicted)
        session['my_var']  = list1

        return render_template('predict.html', prediction_text='Congrats!! you are eligible for the loan')

