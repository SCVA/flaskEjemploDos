import os

from pathlib import Path
import pandas as pd
import numpy as np

import base64
from io import BytesIO
from matplotlib.figure import Figure

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)

app = Flask(__name__)

app_dir = Path(__file__).parent

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))

def obtenerScraping():
    Xt = pd.read_csv(app_dir / "DatosScrapping.csv")
    Xt = Xt.to_numpy()
    Xt = Xt[:, np.newaxis, 1]
    return Xt

@app.route("/prueba")
def prueba():

    Xt = obtenerScraping()

    """
    Cargar mi modelo
    """
    from joblib import load
    with open(app_dir / "modelo_entrenado.pkl", "rb") as f:
        reg = load(f)
    """
    Utilizar mi modelo
    """
    result = reg.predict(Xt)
    resultDF = pd.DataFrame(result,columns=['variable'])
    
    # Generate the figure **without using pyplot**.
    fig = Figure()
    ax = fig.subplots()
    ax.plot(resultDF["variable"])
    ax.set_xlabel('variable')
    ax.set_ylabel("values")
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=8000)
