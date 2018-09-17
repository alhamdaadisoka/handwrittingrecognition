import os
import string
from random import *
from flask import Flask, flash, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from lib.predict import HandwrittingRecognizer as Recognizer

APP_URL = "http://localhost:5000/"
UPLOAD_FOLDER = './static/file'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp', 'gif'])

app = Flask(__name__)
app.secret_key = 'IT TELKOM PURWOKERTO'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#Load recognizer library
recognizer = Recognizer()

#==============================================================================
#FUNGSI-FUNGSI
#==============================================================================
def cek_ekstensi(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_ekstensi(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower()

def random_string():
    min_char = 8
    max_char = 12
    allchar = string.ascii_letters + string.digits
    return "".join(choice(allchar) for x in range(randint(min_char, max_char)))

def generate_new_filename(filename):
    ekstensi = get_ekstensi(filename)
    nama_baru = random_string() + "." + ekstensi

    return nama_baru

#==============================================================================
#FLASK ROUTE
#==============================================================================
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/uploader', methods=['GET','POST'])
def upload():
    if request.method == 'POST':

        # cek apakah ada ada form untuk file
        if 'file' not in request.files:
            flash("Tidak ada file yang diupload")
            return redirect(url_for('home'))

        f = request.files['file']

        #cek apakah gambar belum dipilih
        if f.filename=='':
            flash("Tidak ada file yang diupload")
            return redirect(url_for('home'))

        #cek ekstensi file terlebih dahulu
        if f and cek_ekstensi(f.filename):
            filename = secure_filename(f.filename)
            filename = generate_new_filename(filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(filepath)

            hasil = recognizer.predict(filepath)

            return render_template('result.html', filename=filename, hasil=hasil)
        else:
            flash("Ekstensi file tidak diperbolehkan")
            return redirect(url_for('home'))

    else:
        return redirect(url_for('home'))


#==============================================================================
#FLASK ROUTE FOR API (LIKE ISOMNIA, POSTMAN)
#==============================================================================

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        # cek apakah ada ada form untuk file
        if 'file' not in request.files:
            result = {}
            result['message'] = "No file part in request"
            result['status'] = "fail"
            return jsonify(result)

        f = request.files['file']

        #cek apakah gambar belum dipilih
        if f.filename=='':
            result = {}
            result['message'] = "No file to upload"
            result['status'] = "fail"
            return jsonify(result)

        #cek ekstensi file terlebih dahulu
        if f and cek_ekstensi(f.filename):
            filename = secure_filename(f.filename)
            filename = generate_new_filename(filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(filepath)

            #lakukan prediksi
            hasil = recognizer.predict(filepath)

            result = {}
            result['image'] = APP_URL + "static/file/" + filename
            result['result'] = hasil
            result['status'] = "ok"
            return jsonify(result)

        else:
            result = {}
            result['message'] = "File extension is not allowed"
            result['status'] = "fail"
            return jsonify(result)

    else:
        result = {}
        result['message'] = "Method GET not allowed"
        result['status'] = "fail"
        return jsonify(result)

#==============================================================================

if __name__=='__main__':
    app.run()
