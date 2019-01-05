#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

def preprocess(img_data):
    ''' 画像データのスケーリング/正規化 '''
    mean_vec = np.array([0.485, 0.456, 0.406])[::-1]
    stddev_vec = np.array([0.229, 0.224, 0.225])[::-1]
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[2]):
        # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[:,:,i] = (img_data[:,:,i]/255 - mean_vec[i]) / stddev_vec[i]

    return norm_img_data

if __name__ == '__main__':
    try:
        ## 画像ファイルの読み込み
        file_name = 'beer.jpg'
        input_image: int = cv2.imread(file_name)
        if input_image is not None:
            ## 画像のリサイズ
            resized = cv2.resize(input_image, (224, 224))
            ## 画像のスケーリング/正規化
            preprocessed = preprocess(resized)
            ## Blob形式に変換(行列形状の変換)
            blob = cv2.dnn.blobFromImage(preprocessed)
            print(blob.shape)
            ## ONNXファイルの読み込み
            #model_file = 'resnet50/model.onnx'
            model_file = 'shufflenet/model.onnx'
            net = cv2.dnn.readNetFromONNX(model_file)
            ## 入力画像データの指定
            net.setInput(blob)
            ## フォワードパス(順伝播)の計算 & 不要な次元の削除
            pred = np.squeeze(net.forward())
            print(pred.shape)
            print(sum(pred))
            ## ImageNet(ILSVRC2012)のカテゴリ定義ファイルの読み込み
            rows = open("synset.txt").read().strip().split("\n")
            classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
            ## 推論結果から信頼度の高い順にソートして上位5件のカテゴリ出力
            indexes = np.argsort(pred)[::-1][:5]
            for i in indexes:
                text = "{}: {:.2f}%".format(classes[i], pred[i] * 100)
                print(text)

        else:
            print('can\'t read image')

    except cv2.error as e:
        print(e)
