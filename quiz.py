from flask import Flask, request

app = Flask(__name__)

# Home page with multiple quiz questions
@app.route('/')
def quiz():
    return '''
        <h2>Simple Quiz App</h2>
        <form action="/result" method="post">
            
            <p>Q1: What is the capital of France?</p>
            <input type="radio" name="q1" value="Paris"> Paris<br>
            <input type="radio" name="q1" value="London"> London<br>
            <input type="radio" name="q1" value="Berlin"> Berlin<br>
            <input type="radio" name="q1" value="Rome"> Rome<br><br>

            <p>Q2: Which planet is known as the Red Planet?</p>
            <input type="radio" name="q2" value="Mars"> Mars<br>
            <input type="radio" name="q2" value="Venus"> Venus<br>
            <input type="radio" name="q2" value="Jupiter"> Jupiter<br>
            <input type="radio" name="q2" value="Saturn"> Saturn<br><br>

            <p>Q3: Who developed the theory of relativity?</p>
            <input type="radio" name="q3" value="Einstein"> Albert Einstein<br>
            <input type="radio" name="q3" value="Newton"> Isaac Newton<br>
            <input type="radio" name="q3" value="Galileo"> Galileo Galilei<br>
            <input type="radio" name="q3" value="Tesla"> Nikola Tesla<br><br>

            <input type="submit" value="Submit Quiz">
        </form>
    '''

# Result page with score
@app.route('/result', methods=['POST'])
def result():
    score = 0

    # Correct answers
    answers = {
        "q1": "Paris",
        "q2": "Mars",
        "q3": "Einstein"
    }

    # Calculate score
    for q, correct in answers.items():
        if request.form.get(q) == correct:
            score += 1

    return f"<h2>Your Score: {score}/3</h2>"

if __name__ == '__main__':
    app.run(debug=True)
