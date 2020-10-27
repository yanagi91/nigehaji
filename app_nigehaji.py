import os, re
from flask import Flask, render_template, request, url_for, send_from_directory, redirect, flash
from werkzeug.utils import secure_filename
from pred2 import start_detect


app = Flask(__name__)

UPLOAD_FOLDER = './static/uploads' # 入力画像の保存場所
ALLOWED_EXTENSIONS = set(['jpg','png']) # 拡張子の設定 ここで設定したものしか読み込まない
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

@app.context_processor
def override_url_for():
    """staticの画像の更新用"""
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    # 判定後の画像の保存を上書きしているためhtmlの画像を更新する処理
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
        # ファイルのチェック
        if 'file' not in request.files:
            flash('ファイルがありません', 'error')
            return redirect('/')
        file = request.files['file']
        print(file)
        if file.filename == '':            
            flash('ファイルがありません', 'error')
            return redirect('/')
        
        # ファイルがあり拡張子が対応しているときの処理
        if file and allwed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) # パスを指定し画像を保存

        select_name = request.form['select_name'] # 選択した人物のインデックスを取得
        #print(select_name)
        file_address = 'static/uploads/{}'.format(filename) # 保存した画像のアドレス
        
        predict_name, predict_enname, rate = start_detect(file_address, select_name) # 画像判定

        # 判定後の処理
        if predict_name != None:
            # 選択した人物が画像から見つかった時の処理
            flash(predict_name, 'success')
            flash(rate + '%', 'success')
            dst_img = 'dst/opencv_face_detect_rectangle1.jpg' # 判定後の画像のパス
            return render_template('index.html', dst_img=dst_img)
        else:
            # 選択した人物が画像から見つからなかった時の処理
            flash('検出できませんでした', 'error')

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
