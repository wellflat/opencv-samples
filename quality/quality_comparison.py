#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import sys

if __name__ == '__main__':
    input_image = cv2.imread('images/einstein.png')
    target_files = ['images/einstein.png','images/contrast.png','images/blur.png','images/jpg.png']
    target_images = [cv2.imread(file) for file in target_files]
    mse = cv2.quality.QualityMSE_create(input_image)
    psnr = cv2.quality.QualityPSNR_create(input_image)
    ssim = cv2.quality.QualitySSIM_create(input_image)
    model_path = "./brisque_model_live.yml"
    range_path = "./brisque_range_live.yml"
    brisque = cv2.quality.QualityBRISQUE_create(model_path, range_path)

    for file, image in zip(target_files, target_images):
        mse_score = mse.compute(image)
        psnr_score = psnr.compute(image)
        ssim_score = ssim.compute(image)
        brisque_score = brisque.compute([image])
        print(file,mse_score[0],psnr_score[0],ssim_score[0],brisque_score[0])
