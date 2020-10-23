import os
import sys
import cv2
import argparse
import glob
import facerecg_image_comn as fic

# 顔画像を保存
def save_image(src_image, face_list, fn):
    # 顔だけ切り出して保存
    cnt = 1; # １枚の元画像から検知した顔の数
    for rect in face_list:
        x = rect[0]
        y = rect[1]
        width = rect[2]
        height = rect[3]
        dst = src_image[y:y + height, x:x + width]

        # 書き込み先ディレクトリの存在チェック
        save_dir = os.path.join(args.outdir, os.path.basename(os.path.dirname(fn)))
        if not os.path.isdir(save_dir):
            print('Create save_dir:' + save_dir)
            os.mkdir(save_dir)

        # 顔検知結果の保存
        ext_str = os.path.splitext(fn)[1]
        save_path = os.path.join(save_dir, 'out_' + os.path.basename(fn).replace(ext_str, '_' + '{:0=3}'.format(cnt) + ext_str))
        dummy = fic.imwrite(save_path, dst)
        # plt.show(plt.imshow(np.asarray(Image.open(save_path))))
        # print(cnt)
        cnt += 1

# 顔検知処理メイン
def main(args):
    for fn in glob.iglob(args.indir + "/**", recursive=True):
        # ファイル種別チェック(拡張子jpg/pngのみ対象とする)
        if ('.jpg' in fn) or ('.png' in fn):
            # 画像の読み込み
            print('Raeding...:' + fn)
            src_image = fic.imread(os.path.join(fn))

            # 顔認識用特徴量ファイルを読み込む --- （カスケードファイルのパスを指定）
            cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

            # 顔検知の実行
            face_list = cascade.detectMultiScale(src_image, minNeighbors=3, minSize=(30,30))

            # 認識した顔画像の保存
            save_image(src_image, face_list, fn)

def arg_parse():
    parser = argparse.ArgumentParser(description='Options for scraping Google images')
    parser.add_argument('-i', '--indir', default='./images',
                        type=str, help='input directory path')
    parser.add_argument('-o', '--outdir', default='./facedtct', 
                        type=str, help='output directory path')
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    main(args)