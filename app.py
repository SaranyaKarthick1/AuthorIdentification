
from flask import Flask, request,render_template,url_for
from bs4 import BeautifulSoup as b
import requests
try: 
    from googlesearch import search 
except ImportError:  
    print("No module named 'google' found, pip install google") 
    
from keras.models import load_model    
import pickle 
import tensorflow as tf
graph = tf.get_default_graph()
with open(r'CountVectorizer','rb') as file:
    cv=pickle.load(file)
cla = load_model('author.h5')
cla.compile(optimizer='adam',loss='categorical_crossentropy')    

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/author_identification', methods = ['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        con = request.form['search']
        topic = request.form['name']
        if con.lower()=='predict':
            topic=cv.transform([topic])
            print("\n"+str(topic.shape)+"\n")
            with graph.as_default():
                y_pred = cla.predict_classes(topic)
                print("pred is "+str(y_pred))
            if(y_pred==[0]):
                y_p="EAP"
            elif(y_pred==[1]):
                y_p="HPL"
            else:
                y_p="MWS"
        elif con.lower()=='find by book name':
            html=requests.get("https://www.google.com/search?q="+topic+" book author name").text
            soup=b(html,"lxml")
            l=soup.find_all('a')
            for i in l:
                if i.text.lower()=='wikipedia':
                    y_p=i.get('href').split('&')[0].rsplit('/',1)[1]
        else:
            html=requests.get("https://www.google.com/search?q="+topic).text
            soup=b(html,"html.parser")
            l=soup.find_all('a')
            for i in l:
                if "/url?q=" in i.get("href"):
                    if "https" in i.text.lower():
                        y_p=i.text.split("https")[0]
                        break
                    elif "www" in i.text.lower():
                        y_p=i.text.split("www")[0]
                        break
                    else:
                        y_p=i.text
                        break
        return render_template('index.html',author = y_p)
if __name__ == '__main__':
    app.run(host = 'localhost', debug = True , threaded = False)
    
