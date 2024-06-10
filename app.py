from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo
modelo_rf = joblib.load('modelo_rf.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def realizar_prediccion():
    try:
        # Obtener los datos del POST request
        Credit_History = float(request.form['Credit_History'])
        ApplicantIncome = float(request.form['ApplicantIncome'])
        LoanAmount = float(request.form['LoanAmount'])
        CoapplicantIncome = float(request.form['CoapplicantIncome'])
        Dependents = float(request.form['Dependents'])
    
        
        # Crear un DataFrame con los datos recibidos
        nuevo_registro_df = pd.DataFrame([[Credit_History, ApplicantIncome, LoanAmount, CoapplicantIncome, Dependents]], columns=['Credit_History', 'ApplicantIncome', 'LoanAmount', 'CoapplicantIncome', 'Dependents'])
        app.logger.debug(f'DataFrame creado: {nuevo_registro_df}')
        # Realizar la predicción
        prediccion = modelo_rf.predict(nuevo_registro_df)
        
        app.logger.debug(f'Predicción: {prediccion[0]}')

        return jsonify({'Predicción': int(prediccion[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.debug = True
    app.run()
