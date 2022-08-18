from unittest import result
from flask import Flask , render_template , redirect , request 
from Caption_it import caption_the_image

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/' , methods = ['POST'])
def marks():
    if request.method == 'POST':
        f = request.files['user_file']
        path = "./static/" + f.filename
        f.save(path)
        caption = caption_the_image(path)
        result_dic = {
            'caption': caption , 
            'image' : path 
        }
    return render_template('index.html' , result = result_dic)


if __name__ == "__main__":
    app.run(debug = True)
