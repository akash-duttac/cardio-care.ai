from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.impute import SimpleImputer

# Load the models
lr_file = 'logistic_regression.model'
nb_file = 'naive_bayes.model'
knn_file = 'knn.model'
dt_file = 'dt.model'
svm_file = 'svm.model'

# model = pickle.load(open(filename, 'rb'))

logistic_regression_model = pickle.load(open(lr_file, 'rb'))
naive_bayes_model = pickle.load(open(nb_file, 'rb'))
knn_model = pickle.load(open(knn_file, 'rb'))
dt_model = pickle.load(open(dt_file, 'rb'))
svm_model = pickle.load(open(svm_file, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Validate and clean inputs
            age = request.form.get('age')
            if not age or not age.isdigit() or int(age) <= 0:
                raise ValueError("Invalid age")
            age = int(age)

            sex = request.form.get('sex')
            if sex not in ['0', '1']:
                raise ValueError("Invalid sex")
            sex = int(sex)

            chest_pain_type = request.form.get('chest_pain_type')
            if chest_pain_type not in ['1', '2', '3', '4']:
                raise ValueError("Invalid chest pain type")
            chest_pain_type = int(chest_pain_type)

            resting_blood_pressure = request.form.get('resting_blood_pressure')
            if not resting_blood_pressure or not resting_blood_pressure.isdigit() or int(resting_blood_pressure) <= 0:
                raise ValueError("Invalid resting blood pressure")
            resting_blood_pressure = int(resting_blood_pressure)

            cholestrol = request.form.get('cholestrol')
            if not cholestrol or not cholestrol.isdigit() or int(cholestrol) <= 0:
                raise ValueError("Invalid cholestrol")
            cholestrol = int(cholestrol)

            fasting_blood_sugar = request.form.get('fasting_blood_sugar')
            if fasting_blood_sugar not in ['0', '1']:
                raise ValueError("Invalid fasting blood sugar")
            fasting_blood_sugar = int(fasting_blood_sugar)

            resting_ecg = request.form.get('resting_ecg')
            if resting_ecg not in ['0', '1', '2']:
                raise ValueError("Invalid resting ECG")
            resting_ecg = int(resting_ecg)

            thalach = request.form.get('thalach')
            if not thalach or not thalach.isdigit() or int(thalach) <= 0:
                raise ValueError("Invalid thalach")
            thalach = int(thalach)

            exercise_angina = request.form.get('exercise_angina')
            if exercise_angina not in ['0', '1']:
                raise ValueError("Invalid exercise angina")
            exercise_angina = int(exercise_angina)

            oldpeak = request.form.get('oldpeak')
            if not oldpeak or not is_valid_float(oldpeak):
                raise ValueError("Invalid oldpeak")
            oldpeak = float(oldpeak)

            ST_slope = request.form.get('ST_slope')
            if ST_slope not in ['0', '1', '2']:
                raise ValueError("Invalid ST slope")
            ST_slope = int(ST_slope)

            ca = request.form.get('sex')
            if ca not in ['0', '1']:
                raise ValueError("Invalid sex")
            ca = int(ca)

            thal = request.form.get('thal')
            if thal not in ['0', '1', '2', '3']:
                raise ValueError("Invalid thal value")
            thal = int(thal)

            # Prepare the data for prediction
            data = np.array([[age, sex, chest_pain_type, resting_blood_pressure, cholestrol,
                              fasting_blood_sugar, resting_ecg, thalach, exercise_angina,
                              oldpeak, ST_slope, ca, thal]])

            # Handle any missing values with imputer before prediction
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            data_imputed = imputer.fit_transform(data)

            # Make prediction using the models
            # my_prediction = model.predict(data_imputed)
            lr_prediction = logistic_regression_model.predict(data_imputed)
            nb_prediction = naive_bayes_model.predict(data_imputed)
            knn_prediction = knn_model.predict(data_imputed)
            dt_prediction = dt_model.predict(data_imputed)
            svm_prediction = svm_model.predict(data_imputed)

            return render_template('result.html', lr_prediction=lr_prediction[0], nb_prediction=nb_prediction[0],
                                   knn_prediction=knn_prediction[0], dt_prediction=dt_prediction[0], svm_prediction=svm_prediction[0])

        except ValueError as e:
            # Handle value errors due to invalid input
            return f"Error in input data: {str(e)}"

        except KeyError as e:
            # Handle the case where a required field is not found
            return f"Missing input data: {str(e)}"

# Helper function to check valid float
def is_valid_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

if __name__ == '__main__':
    app.run(debug=True)

