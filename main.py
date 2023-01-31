from flask import Flask
from app import views

app = Flask(__name__)

app.add_url_rule('/base','base',views.base)
app.add_url_rule('/','index',views.index)
app.add_url_rule('/faceApp','faceApp',views.faceApp)
app.add_url_rule('/gender','gender',views.classify,methods=['POST','GET'])

if __name__ == '__main__':
    app.run(debug=True)