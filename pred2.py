  
# -*- coding: utf-8 -*-
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model


input_file = 'test/sample10.jpg'
select_name = '新垣結衣'


def detect_face(image, input_file, select_name):
    print(image.shape)
    #opencvを使って顔抽出
    # image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ##ローカルの場合(/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml)
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    # 顔認識の実行
    face_list=cascade.detectMultiScale(image, minNeighbors=3, minSize=(30,30))
    face = []
    #顔が１つ以上検出された時
    if len(face_list) > 0:
        face_info = []
        a = 0
        for rect in face_list:
            x = rect[0]
            y = rect[1]
            width = rect[2]
            height = rect[3]
            face.append([x,y,width,height])
            face_size = width * height
            img = image[y:y + height, x:x + width]
            face_info.append([face_size, img, rect[0:2], rect[2:4], x, y, height])
            a += 1
            # x,y,width,height=rect

            # img = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            if image.shape[0]<64:
                print("too small")
                continue
        #print(face_info)
        face_info.sort(reverse = True, key=itemgetter(0))
        #print('re:' + str(face_info))
        a -= 1
        for x, y, w, h in face_list:
            predict_img = face_info[a][1]
            print(face_info[a][2:4])
            img = cv2.resize(predict_img,(128, 128))
            img = np.expand_dims(img,axis=0)
            idx, predict_name, predict_enname, rate = detect_who(img) # 抜き出した顔の判定
            if int(select_name) == int(idx):
                result_image = cv2.imread(input_file)
                # 顔座標の取得
                x = int(face_info[a][4])
                y = int(face_info[a][5])
                h = int(face_info[a][6])
                # 顔のある場所を四角で囲う
                cv2.rectangle(result_image, (x, y), (x+h, y+h),(255, 0, 0), 2)
                # 画像の保存
                cv2.imwrite('static/dst/opencv_face_detect_rectangle1.jpg', result_image)
                return predict_name, predict_enname, rate
            else:
                # 選択した人物が見つからない場合の処理
                predict_name = None
            a -= 1

        return predict_name, predict_enname, rate
    # 顔が検出されなかった時
    else:
        print("no face")
        return None, None, None


def detect_who(img):
    # 予測
    name=""
    model = load_model('vgg_model_nigehaji.h5')

    predict = model.predict(img)
    for i, pre in enumerate(predict):
        idx = np.argmax(pre) # 確率の一番高いインデックスの検索
        rate = pre[idx] * 100 # '%'への変換
        rate = str(round(rate, 1)) # 少数第一位まで
        if idx == 0:
            name="新垣結衣"
            en_name="Aragaki Yui"
        elif idx == 1:
            name = "星野源"
            en_name = "Hoshino Gen"
        elif idx == 2:
            name = "真野恵里菜"
            en_name = "Mano Erina"
        elif idx == 3:
            name = "藤井隆"
            en_name = "Huji Takashi"
        elif idx == 4:
            name = "石田ゆり子"
            en_name = "Ishi Yuriko"
        elif idx == 5:
            name = "成田凌"
            en_name = "Narita ryo"
        elif idx == 6:
            name = "古田新太"
            en_name = "Huruta Arata"
        elif idx == 7:
            name = "大谷亮平"
            en_name = "Otani ryohei"
        elif idx == 8:
            name = "宇梶剛士"
            en_name = "Ukaji Takeshi"
        elif idx == 9:
            name = "冨田靖子"
            en_name = "Tomita Yasuko"
        elif idx == 10:
            name = "山賀琴子"
            en_name = "Yamaga Kotoko"
        elif idx == 11:
            name = "古舘寛治"
            en_name = "Hurutate kanji"

    return idx, name, en_name, rate


def start_detect(input_file, select_name):
    global img_file
    img_file = input_file
    model = load_model('vgg_model_nigehaji.h5')

    image = cv2.imread(img_file) # 画像データの読み込み
    print(img_file)
    # 画像が読み込めなかった時の処理
    if image is None:
        print("Not open:")
        
    # 色データの並び替え
    b,g,r = cv2.split(image)
    image = cv2.merge([r,g,b])
    
    # 画像データからの判定処理 predict_name(名前), predict_enname(name), rate(本人の確率)
    predict_name, predict_enname, rate = detect_face(image, input_file, select_name)
    return predict_name, predict_enname, rate


if __name__ == '__main__':
    predict_name, predict_enname, rate = start_detect(input_file, select_name)
    print(predict_name + ':' + predict_enname + ':' + rate)
