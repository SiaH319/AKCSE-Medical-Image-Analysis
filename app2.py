from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("scraping.html")


@app.route('/', methods=['POST'])
def getvalue():
    test = request.form('test')
    print(test)
    return render_template("pass.html", t=test)


if __name__ == '__main__':
