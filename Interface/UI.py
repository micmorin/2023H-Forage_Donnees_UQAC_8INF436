from os import path
from flask import Flask, request, render_template, url_for,redirect, flash
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, recall_score
from sklearn.model_selection import train_test_split

from Dimensions.dimensions import reductionDeDimension
from Preparation.donnees import preparationDesDonnees

import numpy as np
import pandas as pd

def interfaceUtilisateur(modele1, modele2, modele3, donnees, verbose):
  # Begin Flask App
    app = Flask(__name__)
    app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

    # Index.html
    @app.route("/", methods=['GET', 'POST'])
    def index():
        # If GET, we return index.html
        if request.method == 'GET': 
            return render_template('index.html')
        
        # If POST, we process data and return index.html
        else:
            try:
                uploaded_file = request.files['files']

                # If there is a file on the request
                if uploaded_file.filename != '':

                    # If NOT csv File
                    if path.splitext(uploaded_file.filename)[1] not in ['.csv']:
                        flash("Nous n'acceptons que les fichiers .csv")
                        return redirect(url_for('index'))
                    
                    # If csv file
                    else:
                        return render_template('index.html', 
                                               results=getResults(pd.read_csv(uploaded_file.stream),[modele1, modele2, modele3]))
                
                #If no file on request, check table

                # If no file or table data
                else:
                    flash("Vous devez choisir un fichier avant de televerser")
                    return redirect(url_for('index'))
                
            except Exception as e:
                print(e)
                return render_template('index.html')
            

    # Metrics.html
    @app.route("/metrics", methods=['GET'])        
    def metrics():
        X_train, X_test, y_train, y_test = train_test_split(donnees.drop('class_revenue', axis=1), donnees['class_revenue'], test_size=0.3)
        array = getMetrics([modele1, modele2, modele3], X_test, y_test)
        return render_template('metrics.html', array=array)

    # Begin Flask Server
    app.run(host='127.0.0.1',port=80)




def getMetrics(models, x_test, y_test):
    result = []
    cols = []
    index = ["Accuracy","Brier Loss","F1 Score","Recall"]

    # Headers as model names
    for m in models:
        cols.append(type(m).__name__)

    # Measure header at row level
    for i in range(4):
        result.append([])
        # Get metric for each model
        for m in models:
            try:
                if i == 0:   result[i].append(round(accuracy_score(y_test, m.predict(x_test))*100,2))
                elif i == 1: result[i].append(round(brier_score_loss(y_test, np.amax(m.predict_proba(x_test), axis=1))*100,2))
                elif i == 2: result[i].append(round(f1_score(y_test, m.predict(x_test))*100,2))
                elif i == 3: result[i].append(round(recall_score(y_test, m.predict(x_test))*100,2))
                else: result[i].append(111)               
            except:
                # For extremelyFastDecisionTree
                try: 
                    if i == 0:   result[i].append(round(accuracy_score(y_test.to_numpy(), m.predict(x_test.to_numpy()))*100,2))
                    elif i == 1: result[i].append(round(brier_score_loss(y_test.to_numpy(), np.amax(m.predict_proba(x_test.to_numpy()), axis=1))*100,2))
                    elif i == 2: result[i].append(round(f1_score(y_test.to_numpy(), m.predict(x_test.to_numpy()))*100,2))
                    elif i == 3: result[i].append(round(recall_score(y_test.to_numpy(), m.predict(x_test.to_numpy()))*100,2))
                    else: result[i].append(111) 
                except Exception as e:
                    print(e)              
                    result[i].append(0)

    return pd.DataFrame(result, index=index, columns=cols).to_html(classes='table table-striped table-hover text-center')

def getResults(df, models):
    df = preparationDesDonnees(df, verbose = 0)
    df = reductionDeDimension(df, verbose = 0)
    result = []
    cols = ["True_Y"]

    for m in models:
        cols.append(type(m).__name__)

    for i in range(df.shape[0]):
        result.append([df.iloc[i,-1]])
        for m in models:
            try:
                result[i].append(m.predict([df.iloc[i,:-1]])[0])            
            except:
                # For extremelyFastDecisionTree
                try: 
                   result[i].append(m.predict(df.iloc[i,:-1].to_numpy())[0])  
                except Exception as e:
                    print(e)              
                    result[i].append(111)

    t = pd.DataFrame(result,columns=cols)

    return t.to_html(classes='table table-striped table-hover text-center table-responsive', justify='center')