# app.py

from flask import Flask, render_template, request
from predict import PredictPipeline # Correct absolute import
from student_performance_project.predict import PredictPipeline


app = Flask(__name__)

# Initialize the ML pipeline ONCE when the app starts
try:
    pipeline = PredictPipeline()
    print("Web application started and pipeline loaded successfully.")
except FileNotFoundError:
    print("ERROR: Model files missing. Cannot run web app. Run model_train.py first.")
    exit()

@app.route('/')
def index():
    # Renders the HTML form
    return render_template('index.html', results=None)

@app.route('/predict', methods=['POST'])
def predict_datapoint():
    # Capture data from the HTML form (ensure 'name' attributes match keys below)
    data = {
        "gender": request.form.get('gender'),
        "race_ethnicity": request.form.get('ethnicity'),
        "parental_level_of_education": request.form.get('parental_education'),
        "lunch": request.form.get('lunch'),
        "test_preparation_course": request.form.get('test_preparation'),
        # Cast scores to floats expected by the model
        "reading_score": float(request.form.get('reading_score')),
        "writing_score": float(request.form.get('writing_score'))
    }
    
    prediction_result = pipeline.predict(data)
    score = prediction_result.get("prediction", "Error")
    status = prediction_result.get("status")
    
    return render_template('index.html', results=score, status=status)

if __name__ == "__main__":
    # Run the web server
    app.run(host="0.0.0.0", port=5000, debug=True)
