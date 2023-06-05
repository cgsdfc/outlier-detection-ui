from flask import Flask, render_template, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def calculate():
    expression = request.args["expression"]



if __name__ == "__main__":
    app.run(debug=True)
