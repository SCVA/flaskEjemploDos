import os

from pathlib import Path
from matplotlib import pyplot as plt
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from joblib import load

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
    url = "https://www.scimagojr.com/journalrank.php?page=2&total_size=29165"
    req = requests.get(url)
    if(req.status_code==200):
      soup = BeautifulSoup(req.text)
      data = soup.find_all("table")[0]
    dataset = pd.read_html(str(data))[0]
    X = dataset['H index'];
    X= X.to_numpy()
    X = X[:, np.newaxis]
    return X

@app.route("/prueba")
def prueba():
    X = obtenerScraping()
    """
    Cargar mi modelo
    """
    with open("model.pkl", "rb") as f:
        reg = load(f)
    """
    Utilizar mi modelo
    """
    result = reg.predict(X)
    resultDF = pd.DataFrame(result,columns=['variable'])
    resultDF["variable"].plot()
    
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
   app.run(host='0.0.0.0', port=80)
