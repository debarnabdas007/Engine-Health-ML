from flask import Flask, request, render_template
from src.pipelines.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/project')
def project():
    return render_template('project.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            s_2 = float(request.form.get('s_2')),
            s_3 = float(request.form.get('s_3')),
            s_4 = float(request.form.get('s_4')),
            s_7 = float(request.form.get('s_7')),
            s_11 = float(request.form.get('s_11')),
            s_12 = float(request.form.get('s_12')),
            s_15 = float(request.form.get('s_15')),
            s_17 = float(request.form.get('s_17')),
            s_20 = float(request.form.get('s_20')),
            s_21 = float(request.form.get('s_21'))
        )
        
        pred_df = data.get_data_as_dataframe()
        print("User Input received...")

        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)
        
        # --- NEW LOGIC: DETERMINE HEALTH STATUS ---
        rul = round(pred[0], 2)
        
        if rul > 150:
            status = "HEALTHY"
            msg = "Optimal Operations"
            color = "success" # Green
            icon = "fa-check-circle"
        elif rul > 50:
            status = "DEGRADATION"
            msg = "Fault Propagation Detected"
            color = "warning" # Yellow/Orange
            icon = "fa-exclamation-triangle"
        else:
            status = "FAILURE IMMINENT"
            msg = "System Termination Risk"
            color = "danger" # Red
            icon = "fa-radiation"

        return render_template('home.html', 
                               results=rul, 
                               status=status, 
                               msg=msg, 
                               color=color, 
                               icon=icon)
    
if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)