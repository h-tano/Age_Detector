
import json
import numpy as np
import os
import model
import boto3
import cv2
from PIL import Image
from io import BytesIO
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
from keras.preprocessing import image

# 自身の名称を app という名前でインスタンス化する
app = Flask(__name__)
app.config['DEBUG'] = True
# 投稿画像の保存先
UPLOAD_FOLDER = './static/images'

root=1

# ルーティング。/にアクセス時
@app.route('/')
def index():
  return render_template('index.html')

# 画像投稿時のアクション
@app.route('/post', methods=['GET','POST'])
def post():
  if request.method == 'POST':
      if not request.files['file'].filename == u'':
          # アップロードされたファイルをローカルフォルダに保存
          f = request.files['file']
          if root == 2:
              img_path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
              f.save(img_path)

          else:
              # ローカルフォルダに保存したファイルをS3に保存
              bucket_name = "engineerproject1"
              s3 = boto3.resource('s3')
              s3c = boto3.client('s3')
              s3c.upload_fileobj(f, bucket_name, 'upload/'+ f.filename)
              bucket = s3.Bucket(bucket_name)
              object = bucket.Object('upload/' + f.filename)
              img_path = BytesIO(object.get()['Body'].read())

          # model.pyへアップロードされた画像を渡す
          Detect_instance = model.Detectors()
          face_img, coordinate, original = Detect_instance.face_detect(img_path)
          age, gender = Detect_instance.age_detect(face_img/255.0)

          if root == 1:
              for i in range(coordinate.shape[0]):
                  # 顔部分を枠で囲む
                  cv2.rectangle(original, tuple(coordinate[i][0:2]),\
                                tuple(coordinate[i][2:4]),\
                                (0, 122, 200), thickness=2)
                  cv2.putText(original, str(age[i][0]), (coordinate[i][0]+20,coordinate[i][1]+20), \
                              cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 4,lineType=8)
              # 顔部分を赤線で囲った画像の保存先
              now_time = datetime.now().strftime("%Y%m%d%H%M%S")
              detected_img_path = './static/images/' + 'detected_'+ now_time +f.filename
              # 顔部分を赤線で囲った画像の保存
              pil_img = Image.fromarray(original)
              pil_img.save(detected_img_path)

              result = [1, detected_img_path, face_img.shape, coordinate.shape]
          else:
              # JSON 作成
              persons = dict()
              for i in range(age[0].shape[0]):
                  contents = dict()
                  coordi = dict()
                  coordi["x1"]=int(coordinate[i][0])
                  coordi["y1"]=int(coordinate[i][1])
                  coordi["x2"]=int(coordinate[i][2])
                  coordi["y2"]=int(coordinate[i][3])
                  contents["age"] = int(age[i][0])
                  contents["coordinate"] = coordi
                  persons["person"+str(i)] = contents

              persons["image"] = f.filename
              result =[2, json.dumps(persons)]

      else:
          result = []

      return render_template('index.html', result=result)
  else:
      # エラーなどでリダイレクトしたい場合
      return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)
