from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input')
def home():
    return render_template('input_form.html')

@app.route('/result', methods=['GET','POST'])
def result():
    if request.method == 'GET':
        return render_template('index.html')
    
    elif request.method == 'POST':
        pregs = float(request.form.get('Pregnancies'))
        gluc = float(request.form.get('Glucose'))
        bp = float(request.form.get('BloodPressure'))
        skin = float(request.form.get('SkinThickness'))
        insulin = float(request.form.get('Insulin'))
        bmi = float(request.form.get('BMI'))
        func = float(request.form.get('DiabetesPedigreeFunction'))
        age = float(request.form.get('Age'))


        input_features = [[pregs, gluc, bp, skin, insulin, bmi, func, age]]
        # print(input_features)
        with open('new_scaler.pkl','rb') as scaler_file:
            scaler=pickle.load(scaler_file)
        input2 = scaler.transform(input_features)
        with open('random_forest_model.pkl','rb') as model_file:
            rf=pickle.load(model_file)
        prediction = rf.predict(input2)
        # print(prediction)
        result = prediction[0]
        return render_template('result.html',res=result)
    else:
        return render_template('index.html')
    
@app.errorhandler(404)
def page_not_found(error):
    return render_template('err404.html'), 404
    

if __name__ == '__main__':
    app.run(debug=True)