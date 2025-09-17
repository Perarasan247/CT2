from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def home():
    return '''
        <h2>Welcome to the Greeting App!</h2>
        <form action="/greet" method="get">
            Enter your name: <input type="text" name="name">
            <input type="submit" value="Greet Me">
        </form>
    '''

@app.route('/greet')
def greet():
    name = request.args.get("name", "Friend")
    return f"<h2>Hello, {name}! 🎉</h2>"

if __name__ == '__main__':
    app.run(debug=True)
