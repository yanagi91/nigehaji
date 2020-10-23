
#========================================
# 逃げ恥メンバーの顔認識
# ５．学習〜モデルの作成
# ※numpy は新しすぎるとNG。1.18→1.16.2にダウングレードした。
# How To
# Exec Command Sample.
#   python3 facerecg5_train.py
# Options
# パラメータなし。
#========================================
from PIL import Image
import math
import matplotlib.pyplot as plt
import sys
import os
import datetime as dt
import argparse
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras import optimizers
import facerecg_image_comn as fic
import glob
import random

#分類対象のカテゴリーを選ぶ
#CATEGORIES = ["新垣結衣","星野源","真野恵里菜","藤井隆","石田ゆり子","成田凌","古田新太","大谷亮平","宇梶剛士","冨田靖子","山賀琴子","古舘寛治"]
# IMAGE_SIZE_X = 128
# IMAGE_SIZE_Y = 128
# nb_classes = len(fic.categories)

# モデル作成
def train():
    # 開始時刻出力
    starttime = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Start:" + starttime)

    #画像データ(.npy)を読み込み
    X_train, X_test, y_train, y_test = np.load(fic.NPY_FILENAME)
    X_train = X_train.astype("float") / 255
    X_test = X_test.astype("float") / 255
    y_train = np_utils.to_categorical(y_train, fic.NB_CLASSES)
    y_test = np_utils.to_categorical(y_test, fic.NB_CLASSES)
    
    # 入力層の設定
    input_tensor = Input(shape=(fic.IMAGE_SIZE, fic.IMAGE_SIZE, 3))
    
    # vgg16の読み込み
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    # 結合層の変更
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(fic.NB_CLASSES, activation='softmax'))
    
    # 変更した部分をvgg16と接続
    vgg_model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

    # vgg16の14層目までのパラメーターの変更を行わないための処理
    for layer in vgg_model.layers[:15]:
        layer.trainable = False
    
    # パラメーター更新の設定
    vgg_model.compile(loss='categorical_crossentropy',
        optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
        metrics=['accuracy'])

    # 学習の実行
    history = vgg_model.fit(X_train, y_train, batch_size=8,epochs=10)

    # 学習結果のテストの実行
    score = vgg_model.evaluate(X_test, y_test)
    print('loss=', score[0])
    print('accuracy=', score[1])

    #モデルを保存
    vgg_model.save(fic.MODEL_FILENAME)


    # 学習結果を描写
    import matplotlib.pyplot as plt

    #acc, val_accのプロット
    plt.plot(history.history["accuracy"], label="acc", ls="-", marker="o")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(loc="best")
    #Final.pngという名前で、結果を保存
    plt.show()

    # 終了時刻出力
    print("Start:" + starttime)
    print("Finished:" + dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# サンプルデータの追加
def add_sample(cat, fname):
    global X, Y
    # データ読込
    img = Image.open(fname)
    img = img.convert("RGB")    # RGB(8bit x 3)に変換
    # サイズ変更したオリジナル画像をリストに追加
    X.append(np.asarray(img))
    Y.append(cat)   # 画像に紐付けるメンバーの名前を登録

# ファイルごとにループ、add_sampleでサンプルに登録
def make_sample(files):
    global X, Y
    X = []; Y = []
    for cat, fname in files:
        add_sample(cat, fname)
        # add_sample(cat, fname, is_train)
    return np.array(X), np.array(Y)


def arg_parse():
    parser = argparse.ArgumentParser(description='image increase.')
    parser.add_argument('-i', '--indir', default='./increased', type=str, help='input directory path')
    return parser.parse_args()


def main():
    # 引数制御
    args = arg_parse()

    allfiles = []
    testfiles = []
    # 全メンバーのリサイズしたファイルの一覧を取得しリストに保存
    for idx, cat in enumerate(fic.CATEGORIES):
        # 非加工ファイル名一覧を取得しリストに追加
        for fn in glob.glob(os.path.join(args.indir, cat, "*_resz.jpg")):
            testfiles.append((idx, fn))
        # 全ファイル名一覧を取得しリストに追加
        for fn in glob.glob(os.path.join(args.indir, cat, "*.jpg")):
            allfiles.append((idx, fn))

    # ファイル名をランダムに並べ変える
    random.shuffle(testfiles)

    # TEST用のデータ件数を算出（全体件数×（１−トレーニングデータ割合））
    th = math.floor(len(testfiles) * (1.0 - fic.TRAIN_DATA_PER))
    # TEST用データファイルのリストを作成
    test_data_files = testfiles[0:th]

    # 全体ファイル一覧からTEST用データの差分をトレーニング用データとする
    train_data_files = list(set(allfiles) - set(test_data_files))

    # Train用、Test用、それぞれサンプルを作成
    X_train, y_train = make_sample(train_data_files)
    X_test, y_test = make_sample(test_data_files)
    xy = (X_train, X_test, y_train, y_test)
    # すべてのデータをファイルに保存
    np.save(fic.NPY_FILENAME, xy)
    print("ok", len(y_train))

    # トレーニング
    train()
    print("train and test finished.")


if __name__ == '__main__':
    main()
