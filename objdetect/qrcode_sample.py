#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

if __name__ == '__main__':
    file_name = 'qrcode_sample.png'
    input_image = cv2.imread(file_name)
    output_image = input_image.copy()
    detector = cv2.QRCodeDetector()
    data, points, straight_qrcode = detector.detectAndDecode(input_image)
    if data:
        print(f'decoded data: {data}')
        for i in range(4):
            cv2.line(output_image, tuple(points[i][0]), tuple(points[(i + 1) % len(points)][0]), (0, 0, 255), 4)
            cv2.imwrite('output.png', output_image)
        print(f'QR code version: {((straight_qrcode.shape[0] - 21) / 4) + 1}')
