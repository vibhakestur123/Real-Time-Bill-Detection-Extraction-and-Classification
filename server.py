from flask import Flask, url_for, send_from_directory, request, jsonify, render_template
import logging
import os
from werkzeug.utils import secure_filename
from image_processing import image_processing
import json
import pickle
import datetime
import pytz
import uuid
from flask import abort

# compare ensemble to each baseline classifier
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from werkzeug.security import generate_password_hash, check_password_hash


from flask import Flask, render_template, url_for, request, session, redirect
from flask_session import Session
from flask_pymongo import PyMongo
import bcrypt


app = Flask(__name__)
file_handler = logging.FileHandler("server.log")
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

app.config['MONGO_DBNAME'] = 'user'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/myDatabase?retryWrites=true&w=majority'
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config['SECRET_KEY'] = "secret"
app.config['SESSION_COOKIE_NAME'] = "my_session"
Session(app)

mongo = PyMongo(app)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = "{}/uploads/".format(PROJECT_HOME)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

mapper = {
    0: 'Stationery',
    1: 'Edibles',
    2: 'Personal Care'
}

categories = list(mapper.values())
tfidf = pickle.load(open('output/tfidf.sav', 'rb'))
model = pickle.load(open('output/finalized_model.sav', 'rb'))


def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath


@app.route('/')
def index():
    if 'username' in session:
        return render_template('upload.html')

    return render_template('landing.html')


@app.errorhandler(401)
def page_not_found(e):
    return render_template("error.html")


@app.route('/login/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        login_user = mongo.db.users.find_one({'name': request.form['username']})
        if login_user:
            if check_password_hash(login_user['password'], request.form['pass']):
                session['username'] = request.form['username']
                print(session)
                return redirect(url_for('home'))
        else:
            abort(401)
    return render_template('index.html')


@app.route('/register/', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        existing_user = mongo.db.users.find_one({'$or': [{'name': request.form['username'], 'e-mail':request.form['email']}, {'name': request.form['username']}, {'e-mail': request.form['email']}]})

        if existing_user is None:
            hashpass = generate_password_hash(request.form['pass'])
            mongo.db.users.insert_one({'e-mail': request.form['email'], 'name': request.form['username'], 'password': hashpass, 'userData': [], 'totalCost': {'personalCare': 0, 'edibles': 0, 'stationery': 0}})
            session['username'] = request.form['username']
            return redirect(url_for('index'))

        return 'That username or email or both already exists!'

    return render_template('register.html')


@app.route("/process", methods=["POST"])
def api_root():
    if 'username' in session:
        app.logger.info(PROJECT_HOME)
        if request.method == "POST" and request.files["image"]:
            app.logger.info(app.config["UPLOAD_FOLDER"])
            img = request.files["image"]
            img_name = secure_filename(img.filename)
            create_new_folder(app.config["UPLOAD_FOLDER"])
            saved_path = os.path.join(app.config["UPLOAD_FOLDER"], img_name).replace("\\", "/")
            app.logger.info("saving {}".format(saved_path))
            img.save(saved_path)
            q = image_processing(saved_path)
            invoiceProducts = q.process_image()
            cost = {}
            cost['Stationery'] = 0
            cost['Edibles'] = 0
            cost['Personal Care'] = 0
            for each in invoiceProducts:
                print(each["item_name"].lower())
                print(tfidf.transform([each["item_name"].lower()]))
                y_pred = model.predict(tfidf.transform([each["item_name"].lower()]))
                each["category"] = mapper[y_pred[0]]
                cost[each['category']] += (each['price']*each['quantity'])
            userInvData = {}
            userInvData['id'] = uuid.uuid4().hex
            userInvData['date'] = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
            userInvData['products'] = invoiceProducts
            mongo.db.users.update_one({'name': session['username']}, {'$push': {'userData': userInvData}})
            mongo.db.users.update_one({'name': session['username']}, {'$inc': {'totalCost.edibles': cost['Edibles'], 'totalCost.personalCare': cost['Personal Care'], 'totalCost.stationery': cost['Stationery']}})
            total = mongo.db.users.find_one({'name': session['username']}, {'_id': 0, 'totalCost': 1})
            data = total['totalCost']
            data['edibles'] = round(data['edibles'], 2)
            data['stationery'] = round(data['stationery'], 2)
            data['personalCare'] = round(data['personalCare'], 2)
            data['uploaded'] = True
            data['total'] = round(data['edibles'] + data['stationery'] + data['personalCare'], 2)
            #data['ediblesPercentage'] = round(data['edibles'] * 100 / data['total'], 2)
            if data['total'] > 0:
                data['ediblesPercentage'] = round(data['edibles'] * 100 / data['total'], 2)
            else:
                data['ediblesPercentage'] = 0
            data['stationeryPercentage'] = round(data['stationery'] * 100 / data['total'], 2)
            data['personalCarePercentage'] = round(data['personalCare'] * 100 / data['total'], 2)
            return render_template('display.html', content=data)
        else:
            return render_template("upload.html")
    else:
        return render_template("index.html")


@app.route("/home", methods=["GET", "POST"])
def home():
    if 'username' in session:
        return render_template("upload.html")
    else:
        return render_template("index.html")


@app.route("/display", methods=["GET", "POST"])
def display():
    if 'username' in session:
        total = mongo.db.users.find_one({'name': session['username']}, {'_id': 0, 'totalCost': 1})
        data = total['totalCost']
        data['edibles'] = round(data['edibles'], 2)
        data['stationery'] = round(data['stationery'], 2)
        data['personalCare'] = round(data['personalCare'], 2)
        data['total'] = round(data['edibles'] + data['stationery'] + data['personalCare'], 2)
        data['ediblesPercentage'] = round(data['edibles'] * 100 / data['total'], 2)
        data['stationeryPercentage'] = round(data['stationery'] * 100 / data['total'], 2)
        data['personalCarePercentage'] = round(data['personalCare'] * 100 / data['total'], 2)
        return render_template('display.html', content=data)
    else:
        return render_template('index.html')


@app.route("/results", methods=["GET", "POST"])
def results():
    if 'username' in session:
        total = mongo.db.users.find_one({'name': session['username']}, {'_id': 0, 'totalCost': 1})
        data = total['totalCost']
        data['edibles'] = round(data['edibles'], 2)
        data['stationery'] = round(data['stationery'], 2)
        data['personalCare'] = round(data['personalCare'], 2)
        data['total'] = round(data['edibles'] + data['stationery'] + data['personalCare'], 2)
        data['ediblesPercentage'] = round(data['edibles'] * 100 / data['total'], 2)
        data['stationeryPercentage'] = round(data['stationery'] * 100 / data['total'], 2)
        data['personalCarePercentage'] = round(data['personalCare'] * 100 / data['total'], 2)
        return render_template('results.html', content=data, username=session['username'])
    else:
        return render_template('index.html')


@app.route("/details", methods=["GET", "POST"])
def details():
    if 'username' in session:
        det = mongo.db.users.find_one({'name': session['username']}, {'_id': 0, 'userData': 1})
        l = []

        for e in det["userData"]:
            for i in e["products"]:
                d = {}
                d["date"] = e["date"].strftime('%Y-%m-%d')
                if (i["item_name"]) == "":
                    continue
                d["item_name"] = i["item_name"]
                if i["category"] == "Edibles":
                    d["category"] = "Groceries"
                elif i["category"] == "Stationery":
                    d["category"] = "Essentials"
                elif i["category"] == "Personal Care":
                    d["category"] = "Beauty Products"
                d["quantity"] = i["quantity"]
                d["price"] = i["price"]
                if (i["item_name"]) != "":
                    l.append(d)
        # print(l)
        return render_template(
            "details.html",
            content=l
        )
    else:
        return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html", isUserAuthenticated="username" in session)


@app.route("/logout")
def logout():
    session.clear()
    return render_template("index.html")


if __name__ == "__main__":
    app.secret_key = 'mysecret'
    app.run(debug=True)
