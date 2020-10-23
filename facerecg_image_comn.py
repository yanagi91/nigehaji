# ======================================
# Class:image_increase
# 画像Class
# ======================================
import os
import cv2
import numpy as np

# 定数
CATEGORIES = ["新垣結衣","星野源","真野恵里菜","藤井隆","石田ゆり子","成田凌","古田新太","大谷亮平","宇梶剛士","冨田靖子","山賀琴子","古舘寛治"]
NB_CLASSES = len(CATEGORIES)
IMAGE_SIZE = 128
TRAIN_DATA_PER = float(0.9)
NPY_FILENAME = './images_obj.npy'
MODEL_FILENAME = 'vgg_model_nigehaji.h5'

#========================================
def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None
#========================================
def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False
#========================================

#
class Image_increase:
  # 画像データ保持配列定義
  img = []  # 元画像データ
  reimg = []  # 修正後画像データ

  # 定数
  BR_UP = 'up'
  BR_DW = 'down'
  # 画像加工モード
  RESZ = 'resz'
  MOZ = 'moz'
  BLUR = 'blur'
  MONO = 'mono'
  BRUP = 'brup'
  BRDN = 'brdn'
  CONT = 'cont'
  REMODE = [MOZ, BLUR, MONO, BRUP, BRDN, CONT]  # resizeは行わない

  # ======================================
  # __init__
  # ctmファイルの読込。DataFrameで保持する。ヘッダなし。
  # ======================================
  def __init__(self, infilepath, rx, ry):
    self.read(infilepath)
    self.resize(rx, ry)

  # ======================================
  # read
  # ======================================
  def read(self, infilepath):
    self.img = imread(infilepath)

  # ======================================
  # out_filename
  # ======================================
  def out_filename(self, orgfilepath, mode):
    # ファイル名部分と拡張子を取得
    filename, ext = os.path.splitext(os.path.basename(orgfilepath))
    # 画像加工の種類を表すmodeを挟んだ文字列を戻す
    return filename + "_" + mode + ext

  # ======================================
  # write
  # ======================================
  def write(self, outfilepath, outfilename):
    # 書き出し
    imwrite(os.path.join(outfilepath, outfilename), self.reimg)

  # ======================================
  # resize
  # ======================================
  def resize(self, rx, ry):
    copy_img = self.img.copy()
    self.img = cv2.resize(copy_img, (rx, ry),
                                      interpolation=cv2.INTER_LINEAR)
    self.reimg = self.img

  # ======================================
  # mozaic(モザイク)
  # ======================================
  def mozaic(self):
    mosaic_pixcel = 4
    # 入力画像のサイズを取得
    org_h, org_w = self.img.shape[:2]
    copy_img = self.img.copy()
    small_img = cv2.resize(
                                copy_img, 
                                (org_h//mosaic_pixcel, org_w//mosaic_pixcel),
                                interpolation=cv2.INTER_NEAREST)
    self.reimg = cv2.resize(small_img, (org_h, org_w),
                                 interpolation=cv2.INTER_NEAREST)

  # ======================================
  # blur(ぼかし)
  # ======================================
  def blur(self):
    copy_img = self.img.copy()
    self.reimg = cv2.GaussianBlur(copy_img, (5, 5), 0, 0)

  # ======================================
  # monochrome(モノクロ)
  # ======================================
  def monochrome(self):
    copy_img = self.img.copy()
    # グレースケールはやり方がいくつかある。
    # self.reimg = cv2.cvtColor(copy_img, cv2.COLOR_BGR2GRAY)  # RGB2〜 でなく BGR2〜 を指定
    self.reimg, _ = cv2.decolor(copy_img)

  # ======================================
  # bright(明暗)
  # mode=up 明るくする
  # mode=down 暗くする
  # ======================================
  def bright(self, mode):
    copy_img = self.img.copy()
    # 明るさの変化量
    change_value = 64
    # HSV色空間に変換(BGR→HSV)
    # H:色相(Hue)、S:彩度(Saturation)、V:明度(Value/Brightness)
    hsv_img = cv2.cvtColor(copy_img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv_img)
    # 明るくする
    if mode == self.BR_UP:
      # 255以上は255に、それ以外はchange_valueを足した値に。
      v[v > 255-change_value] = 255
      v[v <= 255-change_value] += change_value
    # 暗くする
    elif mode == self.BR_DW:
      # 64以下は0に、それ以外はchange_valueを引いた値に。
      v[v < change_value] = 0
      v[v >= change_value] -= change_value

    # チャネルをマージして1つの画像にして、色変換(HSV→BGR)
    self.reimg = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

  # ======================================
  # contrast(コントラスト)
  # 画像をLAB形式へ変換してコントラストをつける
  # ======================================
  def contrast(self):
    copy_img = self.img.copy()
    # CIE1976(L*, a*, b*)色空間(CIELAB)に変換
    # L=明度(0-100), a,b=補色
    l,a,b = cv2.split(cv2.cvtColor(copy_img, cv2.COLOR_BGR2LAB))
    # 明度の調整
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12,12))
    # 新しい明度で画像を生成
    self.reimg = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)

  # ======================================
  # allmodi
  # 全ての種類の画像加工を連続して実施する
  # ======================================
  def allmodi(self, infilepath, outfilepath):
    print("filepath: %s" % infilepath)
    # 加工前の、サイズ変更のみ行った画像の出力
    self.write(outfilepath, self.out_filename(infilepath, self.RESZ))

    # 以下、画像加工処理
    for moditype in self.REMODE:
      print("modetype: %s" % moditype)
      if moditype == self.MOZ:
        self.mozaic()
      elif moditype == self.BLUR:
        self.blur()
      elif moditype == self.MONO:
        self.monochrome()
      elif moditype == self.BRUP:
        self.bright(self.BR_UP)
      elif moditype == self.BRDN:
        self.bright(self.BR_DW)
      elif moditype == self.CONT:
        self.contrast()
      # 加工後のファイル書き出し。出力先はoutfilepath。infilepathはファイル名の取得に使用
      self.write(outfilepath, self.out_filename(infilepath, moditype))