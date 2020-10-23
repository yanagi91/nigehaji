
import os, sys, glob
import argparse
import numpy as np
import cv2
import facerecg_image_comn as fic

# 画像データを読み込む
X = []  # 画像データ
Y = []  # ラベルデータ


def arg_parse():
    parser = argparse.ArgumentParser(description='image increase.')
    parser.add_argument('-i', '--indir', default='./facedtct',
                        type=str, help='input directory path')
    parser.add_argument('-o', '--outdir', default='./increased',
                        type=str, help='output directory path')
    return parser.parse_args()


def main(args):
    allfiles = []
    # メンバー名ごとに処理
    for idx, cat in enumerate(fic.CATEGORIES):
        # ファイル名一覧取得
        print(cat)
        for fn in glob.glob(os.path.join(args.indir, "*" + cat + "*", "*.jpg")):
            # facerecg_image_comn クラスの呼び出しと全画像の一括加工
            images = fic.Image_increase(fn, fic.IMAGE_SIZE, fic.IMAGE_SIZE)
            # ディレクトリが存在しなければ作成する
            if not os.path.isdir(os.path.join(args.outdir, cat)):
                os.mkdir(os.path.join(args.outdir, cat))
                print('outdir:' + os.path.join(args.outdir, cat))
            images.allmodi(fn, os.path.join(args.outdir, cat))
            

if __name__ == '__main__':
    args = arg_parse()
    main(args)
