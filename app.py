from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model files
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('columns.pkl', 'rb') as f:
    columns = pickle.load(f)
with open('label_encoder_name.pkl', 'rb') as f:
    le_name = pickle.load(f)

# Load car names for dropdown
car_names = list(le_name.classes_)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            name_encoded = le_name.transform([request.form['name']])[0]
            year = int(request.form['year'])

            data = {
                'name': name_encoded,
                'year': year,
                'km_driven': int(request.form['km_driven']),
                'mileage': float(request.form['mileage']),
                'engine': float(request.form['engine']),
                'max_power': float(request.form['max_power']),
                'seats': int(request.form['seats']),
                'owner': int(request.form['owner']),
                'fuel_Diesel': int(request.form.get('fuel') == 'Diesel'),
                'fuel_Electric': int(request.form.get('fuel') == 'Electric'),
                'fuel_LPG': int(request.form.get('fuel') == 'LPG'),
                'fuel_Petrol': int(request.form.get('fuel') == 'Petrol'),
                'seller_type_Individual': int(request.form.get('seller_type') == 'Individual'),
                'seller_type_Trustmark Dealer': int(request.form.get('seller_type') == 'Trustmark Dealer'),
                'transmission_Manual': int(request.form.get('transmission') == 'Manual'),
            }

            df_input = pd.DataFrame([data])
            for col in columns:
                if col not in df_input.columns:
                    df_input[col] = 0
            df_input = df_input[columns]

            df_scaled = scaler.transform(df_input)
            prediction = model.predict(df_scaled)[0]
            return render_template('predict.html', prediction=round(prediction, 2), car_names=car_names)
        
        except Exception as e:
            print("❌ Error:", e)
            return render_template('predict.html', prediction="Error occurred", car_names=car_names)

    return render_template('predict.html', prediction=None, car_names=car_names)

if __name__ == '__main__':
    app.run(debug=True)
