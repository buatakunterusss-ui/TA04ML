from flask import Flask, render_template, request
import joblib, json, pandas as pd

app = Flask(__name__)
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')
contoh = json.load(open('contoh.json'))

@app.route('/', methods=['GET', 'POST'])
def index():
    hasil = None
    nilai = {}
    if request.method == 'POST':
        nilai = {c: float(request.form.get(c) or 0) for c in columns}
        X_in = scaler.transform(pd.DataFrame([nilai], columns=columns))
        proba = model.predict_proba(X_in)[0][1]
        label = "Berisiko Fraud" if proba >= 0.5 else "Tidak Berisiko"
        hasil = {'label': label, 'proba': round(proba * 100, 2), 'fraud': proba >= 0.5}
    return render_template('index.html', hasil=hasil, columns=columns,
                           nilai=nilai, contoh=contoh)

if __name__ == '__main__':
    app.run(debug=True)