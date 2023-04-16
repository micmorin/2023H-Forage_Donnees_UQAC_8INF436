from os import path
from sys import stdout
from flask import Flask, request, render_template, url_for,redirect, flash
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, recall_score, confusion_matrix
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
                form = request.form.to_dict(flat=False)

                # File upload
                if len(form) == 1:
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
                    
                    # If no file
                    else:
                        flash("Vous devez choisir un fichier avant de televerser")
                        return redirect(url_for('index'))

                # Table Data Upload    
                else:
                    table = []
                    for row_number in range(1,int(len(form)/8)+1):
                        row = []
                        n = str(row_number)
                        row.append(int(form["age"+n][0]))
                        row.append(int(form["page"+n][0]))
                        row.append(int(form["prix"+n][0]))
                        row.append(form["genre"+n][0])
                        row.append(bool(form["achats"+n][0]))
                        row.append(int(form["pub"+n][0]))
                        row.append(int(form["pays"+n][0]))
                        row.append(int(form["revenue"+n][0]))
                        table.append(row)
                    df = pd.DataFrame(table,columns=['age','pages','first_item_prize','gender','ReBuy','News_click','country', 'revenue'])
                    return render_template('index.html', 
                                                results=getResults(df,[modele1, modele2, modele3]))
                
            except Exception as e:
                print(e)
                return render_template('index.html')
            

    # Metrics.html
    @app.route("/metrics", methods=['GET'])        
    def metrics():
        X_train, X_test, y_train, y_test = train_test_split(donnees.drop('class_revenue', axis=1), donnees['class_revenue'], test_size=0.3)
        array = getMetrics([modele1, modele2, modele3], X_test.to_numpy(), y_test.to_numpy())
        confusionMatrix = getConfusion([modele1, modele2, modele3], X_test.to_numpy(), y_test.to_numpy())
        guesses = getRightGuesses([modele1, modele2, modele3], X_test.to_numpy(), y_test.to_numpy())
        return render_template('metrics.html', array=array, confusionMatrix = confusionMatrix, guesses = guesses)

    # Begin Flask Server
    app.run(host='127.0.0.1',port=82)




def getMetrics(models, x_test, y_test):
    result = []
    cols = []
    index = ["Accuracy","Brier Loss","F1 Score","Recall"]

    # Headers as model names
    for m in models:
        cols.append(type(m).__name__)
    cols.append("Votes (Majority)")

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

    # Mesute the vote of all models
    predAll = models[0].predict(x_test) + models[1].predict(x_test) + models[2].predict(x_test)
    predAll = [1 if x > 1 else 0 for x in predAll]

    # Adding it to the table
    result[0].append(round(accuracy_score(y_test, predAll)*100,2))
    result[1].append(round(brier_score_loss(y_test, predAll)*100,2))
    result[2].append(round(f1_score(y_test, predAll)*100,2))
    result[3].append(round(recall_score(y_test, predAll)*100,2))

    return pd.DataFrame(result, index=index, columns=cols).to_html(classes='table table-striped table-hover text-center').replace('<tr style="text-align: right;">', '<tr>')

def getConfusion(models, x_test, y_test):
    result = [[],[],[],[]]
    cols = []
    index = ['Vrai Positif','Vrai Négatif','Faux Positif','Faux Négatif']

    # Headers as model names
    for m in models:
        cols.append(type(m).__name__)
    cols.append("Votes (Majority)")

    for m in models:
        confusionMatrix = confusion_matrix(y_test, m.predict(x_test))
        result[0].append(confusionMatrix[0][0])
        result[1].append(confusionMatrix[1][1])
        result[2].append(confusionMatrix[0][1])
        result[3].append(confusionMatrix[1][0])

    # Mesute the vote of all models
    predAll = models[0].predict(x_test) + models[1].predict(x_test) + models[2].predict(x_test)
    predAll = [1 if x > 1 else 0 for x in predAll]

    # Adding it to the table
    confusionMatrix = confusion_matrix(y_test, predAll)
    result[0].append(confusionMatrix[0][0])
    result[1].append(confusionMatrix[1][1])
    result[2].append(confusionMatrix[0][1])
    result[3].append(confusionMatrix[1][0])

    return pd.DataFrame(result, index=index, columns=cols).to_html(classes='table table-striped table-hover text-center').replace('<tr style="text-align: right;">', '<tr>')


def getRightGuesses(models, x_test, y_test):
    result = [[]]
    cols = ["0 model","1 model","2 models","3 models"]
    index = ['']

    predAll = models[0].predict(x_test) + models[1].predict(x_test) + models[2].predict(x_test)
    #revert where it's suppose to be 0
    predAll = [3-predAll[i] if y_test[i] == 0 else predAll[i] for i in range(len(predAll))]
    #count the occurences
    for i in range(4):
        result[0].append(predAll.count(i))

    df = pd.DataFrame(result, index=index, columns=cols)
    df.columns.name = 'Guessed right by :'
    return df.to_html(classes='table table-striped table-hover text-center').replace('<tr style="text-align: right;">', '<tr>')


def getResults(df, models):
    if df.shape[0] == 1: 
        df = pd.DataFrame([df.iloc[0].to_list(),[0,0,0,'Masc',False,1,1,1]],columns=df.columns)
        df = preparationDesDonnees(df, verbose = 0)
        df = reductionDeDimension(df, verbose = 0)
        df = df.drop(index=1)

    else:
        df = preparationDesDonnees(df, verbose = 0)
        df = reductionDeDimension(df, verbose = 0)
    result = []
    cols = ["True_Y"]

    for m in models:
        cols.append(type(m).__name__)

    for i in range(df.shape[0]):
        stdout.write("Predicting..."+str(round(i/df.shape[0]*100,2))+'%\r')
        stdout.flush()
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