import os, re
from flask import Flask, render_template, request, url_for, send_from_directory, redirect, flash
from werkzeug.utils import secure_filename
from pred2 import start_detect


app = Flask(__name__)

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = set(['jpg','png'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


def allwed_file(filename):
    # ファイルの拡張子の確認
    # OKなら１、だめなら0
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def uploads_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません', 'error')
            return redirect('/')

        file = request.files['file']
        print(file)
        if file.filename == '':
            flash('ファイルがありません', 'error')
            return redirect('/')
        
        if file and allwed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        select_name = request.form['select_name']
        #print(select_name)
        if os.path.exists('static/dst/opencv_face_detect_rectangle1.jpg'):
            os.remove('static/dst/opencv_face_detect_rectangle1.jpg')
        file_address = 'static/uploads/{}'.format(filename) # 判定画像アドレス
        
        predict_name, predict_enname, rate = start_detect(file_address, select_name) # 画像判定

        if predict_name != None:
            flash(predict_name, 'success')
            flash(rate + '%', 'success')
            dst_img = 'dst/opencv_face_detect_rectangle1.jpg'
            return render_template('index.html', dst_img=dst_img)
        else:
            flash('検出できませんでした', 'error')

    return render_template('index.html')


@app.route('/dst/<filename>')
def uploaded_file(filename):
    return send_from_directory(filename)

if __name__ == '__main__':
    app.run(debug=True)