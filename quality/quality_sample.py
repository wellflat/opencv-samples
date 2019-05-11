#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import sys

if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     print('usage: ./quality_sample.py <image path>')
    #     sys.exit(-1)

    model_path = "./brisque_model_live.yml"
    range_path = "./brisque_range_live.yml"
    input_image = cv2.imread(sys.argv[1])
    quality = cv2.quality.QualityBRISQUE_create(model_path, range_path)
    score = quality.compute([input_image])
    print(score)

    ref_image = cv2.imread(sys.argv[2])
    quality2 = cv2.quality.QualityMSE_create(ref_image)
    score2 = quality2.compute(input_image)
    print(score2[0])
