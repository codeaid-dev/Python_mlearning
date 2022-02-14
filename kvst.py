from flask import Flask, render_template, request, redirect
import os
import pandas as pd
from sklearn import tree
import pickle

app = Flask(__name__)

RESULT = './result.txt'
MODEL = './data/KvsT.pkl'
df,model = None,None

def learn():
    global df,model
    if os.path.exists(MODEL):
        with open(MODEL, 'rb') as f:
            model = pickle.load(f)
    else:
        df = pd.read_csv('./data/KvsT.csv')
        model = tree.DecisionTreeClassifier(random_state = 0)
        xcol = ['身長', '体重', '年代']
        x = df[xcol]
        t = df['派閥']
        model.fit(x,t)
        with open(MODEL, 'wb') as f:
            pickle.dump(model, f)

@app.route('/')
def index():
    result = '入力してください。'
    if os.path.exists(RESULT):
        with open(RESULT, 'r') as f:
            result = f.read()
        result = result.replace('\n', '<br>')
    
    learn()

    return render_template('kvst.html', result=result)

@app.route('/predict', methods=['POST'])
def predict():
    name, height, weight, age = '','','',''
    if 'name' in request.form:
        name = request.form['name']
    if 'height' in request.form:
        height = int(request.form['height'])
    if 'weight' in request.form:
        weight = int(request.form['weight'])
    if 'age' in request.form:
        age = int(request.form['age'])

    result = f'名前：{name}\n身長：{height}\n体重：{weight}\n年代：{age}\n'

    ans = model.predict([[height,weight,age]])
    result += f'あなたは、{ans[0]}派です。\n'

    with open(RESULT, 'w') as f:
        f.write(result)

    return redirect('/')

app.run(port=8000, debug=True)
