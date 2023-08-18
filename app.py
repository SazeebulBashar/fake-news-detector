#Implement all this concept by machine learning with flask

from flask import Flask, escape, request, render_template
import pickle

vector = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("finalized_model.pkl", 'rb'))
model2 = pickle.load(open("model2.pkl", 'rb'))
model3 = pickle.load(open("model3.pkl", 'rb'))
model4 = pickle.load(open("model4.pkl", 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        news = str(request.form['news'])
        print(news)

        predict = model.predict(vector.transform([news]))[0]

        predict2 = model2.predict(vector.transform([news]))[0]
        predict3 = model3.predict(vector.transform([news]))[0]
        predict4 = model4.predict(vector.transform([news]))[0]

    
        return render_template("prediction.html",p1 ="PassiveAggressiveClassifier: " +  predict + " News", 
                                                p2 = "Multinomial NB: " + predict2 + " News", 
                                                p3 = "Support Vector Machine: " + predict3+ " News", 
                                                p4 ="Logistic Regression: " + predict4+ " News")

    else:
        return render_template("prediction.html")


if __name__ == '__main__':
    app.debug = True
    app.run()
