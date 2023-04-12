from flask import Flask, request, render_template, url_for,redirect, flash
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, recall_score
from sklearn.model_selection import train_test_split
from river import metrics
import numpy as np

def interfaceUtilisateur(modele1, modele2, modele3, donnees, verbose):
  # Begin Flask App
    app = Flask(__name__)

    x_train, x_test, y_train, y_test = train_test_split(donnees.iloc[:,:-1], donnees.iloc[:,-1:], test_size=0.25)

    # Index.html
    @app.route("/", methods=['GET', 'POST'])
    def index():
        # If GET, we return index.html
        if request.method == 'GET': 
            return render_template('index.html')
        
        # If POST, the team is selected so redirect to level 1
        else:
            return redirect(url_for('metrics'))

    # Metrics.html
    @app.route("/metrics", methods=['GET'])        
    def metrics():
        array = getMetrics([modele1, modele2, modele3], x_test, y_test)
        return render_template('metrics.html', array=array)

    # Begin Flask Server
    app.run(host='127.0.0.1',port=80)

def getMetrics(models, x_test, y_test):
    result = [[]]
    for m in models:
        result[0].append(type(m).__name__)

    for i in range(1,5):
        if i == 1:   result.append(["Accuracy"])
        elif i == 2: result.append(["Brier Loss"])
        elif i == 3: result.append(["F1 Score"])
        elif i == 4: result.append(["Recall"])
        else: result.append([""])

        for m in models:
            try:
                if i == 1:   result[i].append(round(accuracy_score(y_test, m.predict(x_test))*100,2))
                elif i == 2: result[i].append(round(brier_score_loss(y_test, np.amax(m.predict_proba(x_test), axis=1))*100,2))
                elif i == 3: result[i].append(round(f1_score(y_test, m.predict(x_test))*100,2))
                elif i == 4: result[i].append(round(recall_score(y_test, m.predict(x_test))*100,2))
                else: result[i].append(111)
            except:
                try:
                    if i == 1:   result[i].append(metrics.Accuracy().get())
                    elif i == 2: result[i].append(0.0)
                    elif i == 3: result[i].append(metrics.F1().get())
                    elif i == 4: result[i].append(metrics.Recall().get())
                    else: result[i].append(111)
                except Exception as e:
                    print(e)
                    result[i].append(0)

    return result