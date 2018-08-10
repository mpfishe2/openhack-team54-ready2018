from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
#@app.route('/index')
def hi():
    return(render_template("index.html"))

@app.route("/hello")
def hello():
    return "Hello World"