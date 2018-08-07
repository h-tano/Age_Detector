import numpy as np
import cv2
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from ssd import SSD300
from ssd_utils import BBoxUtility




class Detectors:

    def __init__(self):
        #顔検出モデルと年齢・性別検出モデルを復元
        self.age_detector = load_model("transfer_Xception_29.h5")
        NUM_CLASSES = 2
        input_shape = (300, 300, 3)
        priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
        self.bbox_util = BBoxUtility(NUM_CLASSES, priors)
        self.face_detector = SSD300(input_shape, num_classes=NUM_CLASSES)
        self.face_detector.load_weights('weights.05-3.15.hdf5', by_name=True)


    def age_detect(self, input):
        #先頭にNUMの次元が必要なので追加
        input_add = input
        age_predict = self.age_detector.predict(input_add)
        # 年齢をsigmoidの出力（０〜１）から元に戻す（１を１１６歳にしている）
        age = np.round(age_predict[0]*116).astype(int)
        # 性別
        gender = np.zeros([age_predict[1].shape[0],1],dtype=str)
        for i in range(age_predict[1].shape[0]):
            # 性別は[0.2,0.8]ならF , [0.6,0.4]ならM のように判定
            if 0.5 <= age_predict[1][i][0]:
                gender[i] = 'M'
            else:
                gender[i] = 'F'

        return age, gender #リターンをarray形式で統一


    def face_detect(self, img_path, display=False):

        inputs, images, resize_imgs, bb_coordinate = [], [], [], []

        img = image.load_img(img_path, target_size=(300, 300))
        img = image.img_to_array(img)

        if '/' in img_path:
            img_original = image.load_img(img_path)
            img_original = image.img_to_array(img_original)
        else:
            # s3から取得した場合
            img_original = np.array(image.load_img(img_path))


        images.append(img_original)
        inputs.append(img)
        inputs = preprocess_input(np.array(inputs))

        # predict
        preds = self.face_detector.predict(inputs, batch_size=1, verbose=0)
        results = self.bbox_util.detection_out(preds)

        for i, img in enumerate(images):
            # Parse the outputs.
            det_label = results[i][:, 0]
            det_conf = results[i][:, 1]
            det_xmin = results[i][:, 2]
            det_ymin = results[i][:, 3]
            det_xmax = results[i][:, 4]
            det_ymax = results[i][:, 5]

            top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.3]

            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]

            for i in range(top_conf.shape[0]):
                xmin = int(round(top_xmin[i] * img.shape[1]))
                ymin = int(round(top_ymin[i] * img.shape[0]))
                xmax = int(round(top_xmax[i] * img.shape[1]))
                ymax = int(round(top_ymax[i] * img.shape[0]))
                score = top_conf[i]
                label = int(top_label_indices[i])

                bb_coordinate.append(np.array([xmin, ymin, xmax, ymax]))
                detect_img = img_original[ymin:ymax, xmin: xmax, :]
                detect_img = cv2.resize(detect_img,(200, 200))
                resize_imgs.append(detect_img)

        # リサイズ画像の配列と元画像の位置左上の(x,y), 右下の(x,y)を返す
        return np.array(resize_imgs), np.array(bb_coordinate), img_original
