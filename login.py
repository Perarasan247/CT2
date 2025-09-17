from flask import Flask, request

app = Flask(__name__)

USERNAME = "admin"
PASSWORD = "12345"

@app.route("/")
def home():
    return """
    <h2>Login Page</h2>
    Enter your username and password in the URL like this:<br>
    /login?username=admin&password=12345
    """

@app.route("/login")
def login():
    username = request.args.get("username")
    password = request.args.get("password")

    if not username or not password:
        return "❌ Please provide both username and password in the URL."

    if username == USERNAME and password == PASSWORD:
        return f"✅ Welcome, {username}! You are successfully logged in."
    else:
        return "❌ Invalid username or password. Try again."

if __name__ == "__main__":
    app.run(debug=True)
