from os import path
from flask import Flask, request, render_template, url_for,redirect, flash
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, recall_score
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

def interfaceUtilisateur(modele1, modele2, modele3, donnees, verbose):
  # Begin Flask App
    app = Flask(__name__)
    app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

    x_train, x_test, y_train, y_test = train_test_split(donnees.iloc[:,:-1], donnees.iloc[:,-1:], test_size=0.25)

    # Index.html
    @app.route("/", methods=['GET', 'POST'])
    def index():
        # If GET, we return index.html
        if request.method == 'GET': 
            return render_template('index.html')
        
        # If POST, 
        else:
            try:
                uploaded_file = request.files['files']
                if uploaded_file.filename != '':
                    if path.splitext(uploaded_file.filename)[1] not in ['.csv']:
                        flash("Nous n'acceptons que les fichiers .csv")
                        return redirect(url_for('index'))
                    else:
                        return render_template('index.html')
                else:
                    flash("Vous devez choisir un fichier avant de televerser")
                    return redirect(url_for('index'))
            except Exception as e:
                print(e)
                return render_template('index.html')
            

    # Metrics.html
    @app.route("/metrics", methods=['GET'])        
    def metrics():
        array = getMetrics([modele1, modele2, modele3], x_test, y_test)
        return render_template('metrics.html', array=array)

    # Begin Flask Server
    app.run(host='127.0.0.1',port=80)

def getMetrics(models, x_test, y_test):
    result = [[]]

    # Headers as model names
    for m in models:
        result[0].append(type(m).__name__)

    # Measure header at row level
    for i in range(1,5):
        if i == 1:   result.append(["Accuracy"])
        elif i == 2: result.append(["Brier Loss"])
        elif i == 3: result.append(["F1 Score"])
        elif i == 4: result.append(["Recall"])
        else: result.append([""])

        # Get metric for each model
        for m in models:
            try:
                if i == 1:   result[i].append(round(accuracy_score(y_test, m.predict(x_test))*100,2))
                elif i == 2: result[i].append(round(brier_score_loss(y_test, np.amax(m.predict_proba(x_test), axis=1))*100,2))
                elif i == 3: result[i].append(round(f1_score(y_test, m.predict(x_test))*100,2))
                elif i == 4: result[i].append(round(recall_score(y_test, m.predict(x_test))*100,2))
                else: result[i].append(111)               
            except:
                # For extremelyFastDecisionTree
                try: 
                    if i == 1:   result[i].append(round(accuracy_score(y_test.to_numpy(), m.predict(x_test.to_numpy()))*100,2))
                    elif i == 2: result[i].append(round(brier_score_loss(y_test.to_numpy(), np.amax(m.predict_proba(x_test.to_numpy()), axis=1))*100,2))
                    elif i == 3: result[i].append(round(f1_score(y_test.to_numpy(), m.predict(x_test.to_numpy()))*100,2))
                    elif i == 4: result[i].append(round(recall_score(y_test.to_numpy(), m.predict(x_test.to_numpy()))*100,2))
                    else: result[i].append(111) 
                except Exception as e:
                    print(e)              
                    result[i].append(0)

    return result

def getResults(file):
    t = pd.read_csv(file)
    return t