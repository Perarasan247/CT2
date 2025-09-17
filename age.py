from flask import Flask, request
from datetime import datetime

app = Flask(__name__)

@app.route("/")
def home():
    return """
    Enter your DOB in the URL like this:<br>
    /calculate?dob=1995-08-20
    """

@app.route("/calculate")
def calculate_age():
    dob_str = request.args.get("dob")
    if not dob_str:
        return "❌ Please provide your DOB in the format YYYY-MM-DD"

    try:
        dob = datetime.strptime(dob_str, "%Y-%m-%d")
        today = datetime.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return f"🎉 Your Age is: {age} years"
    except:
        return "❌ Invalid date format. Use YYYY-MM-DD"

if __name__ == "__main__":
    app.run(debug=True)
