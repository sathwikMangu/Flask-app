from flask import Flask, render_template, request
import os
import matplotlib.pyplot as plt


from convert_final import convert
from predict_disease import find_disease


UPLOAD_FOLDER = os.path.join('static')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/display')
def display():
    return render_template('display.html')


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if request.files['audiofile'].filename == '':
            return 'No audio recording is selected'

        if 'audiofile' not in request.files:
            return 'there is no audiofile'
        file = request.files['audiofile']
        path = UPLOAD_FOLDER+"/input.wav"
        # print(path)
        file.save(path)

        convert(path)

        chroma = UPLOAD_FOLDER+"/chroma.jpg"
        mfccs = UPLOAD_FOLDER+"/mfccs.jpg"
        mel = UPLOAD_FOLDER+"/mel.jpg"

        find_disease(path)

        probable_disease = UPLOAD_FOLDER+"/heatmap.jpg"

        return render_template("display.html", chroma=chroma, mel=mel, mfccs=mfccs, heatmap=probable_disease)

    return render_template('del_test.html')


if __name__ == '__main__':
    #app.run(host='0.0.0.0',port=5000)
    # app.run(host='0.0.0.0',port=8080)
    #app.run(debug=False,host='0.0.0.0')
    app.run(debug=True)
