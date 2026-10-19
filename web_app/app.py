import os  # Moved to the very top
import sys
from flask import Flask, render_template, request, session, redirect, url_for
from flask_dance.contrib.google import make_google_blueprint, google

# 1. ALLOW INSECURE TRANSPORT (Must be set before blueprint is used)
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# --- TRY CLEAN IMPORT FIRST ---
try:
    from student_performance_project.predict import predict_score
except ImportError:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOGIC_DIR = os.path.join(BASE_DIR, 'student_performance_project')
    if LOGIC_DIR not in sys.path:
        sys.path.append(LOGIC_DIR)
    try:
        from predict import predict_score
    except ImportError as e:
        raise ImportError(f"Could not import predict_score. Checked: {LOGIC_DIR}") from e

app = Flask(__name__)
app.secret_key = "secret123"

# 2. GOOGLE BLUEPRINT (Changed prefix to /google_login to avoid conflict)
google_bp = make_google_blueprint(
    client_id=os.getenv("GOOGLE_OAUTH_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_OAUTH_CLIENT_SECRET"),
google_bp = make_google_blueprint(
    scope=[
        "openid",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile"
    ],
    
    offline=True,
    redirect_to="google_login_handler" # Change this name
)
app.register_blueprint(google_bp, url_prefix="/google_login")

@app.route('/')
def home():
    return render_template('home.html')

# 3. MANUAL LOGIN (Keep this as /login)
@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    if username == "admin" and password == "1234":
        session['user'] = username
        return redirect(url_for('index'))
    else:
        return render_template('home.html', error="Invalid credentials")

# 4. GOOGLE LOGIN HANDLERs
@app.route('/google_login_handler') # Update this URL
def google_login_handler():         # Update this function name
    if not google.authorized:
        return redirect(url_for("google.login"))
    
    resp = google.get("/oauth2/v2/userinfo")
    if not resp.ok:
        return "Google login failed"

    user_info = resp.json()
    session['user'] = user_info["email"]
    return redirect(url_for('index'))

@app.route('/predict_form')
def index():
    return render_template('index.html', input_data={})

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form.to_dict()
    try:
        input_data["math_score"] = float(input_data.get("math_score") or 0)
        input_data["reading_score"] = float(input_data.get("reading_score") or 0)
        
        result = predict_score(input_data)
        if result.get('status') == 'success':
            return render_template('result.html', prediction=result.get('prediction'), input_data=input_data)
        else:
            return render_template('index.html', error=result.get('status'), input_data=input_data)
    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}", input_data=input_data)
    
@app.route('/analysis')
def show_analysis():
    return render_template('analysis.html')
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)