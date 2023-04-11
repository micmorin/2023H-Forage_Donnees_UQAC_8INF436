from flask import Flask, request, render_template, url_for,redirect, flash

def interfaceUtilisateur(modele1, modele2, modele3, verbose):
  # Begin Flask App
    app = Flask(__name__)

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
        return render_template('metrics.html')

    # Begin Flask Server
    app.run(host='127.0.0.1',port=80)
