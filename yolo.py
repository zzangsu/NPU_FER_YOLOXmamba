import colorsys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
from utils.utils_bbox import decode_outputs, non_max_suppression

import onnxruntime as ort

class YOLO(object):
    def __init__(self, class_path, onnx_path, input_shape):
        self.classes_path = class_path
        self.onnx_path = onnx_path
        self.input_shape = input_shape
        self.letterbox_image = True
        self.confidence=0.5
        self.nms_iou=0.3


        self.class_names, self.num_classes  = get_classes(self.classes_path)
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()    
        
    def generate(self, onnx=False):
        self.net    = ort.InferenceSession(self.onnx_path)
        print('{} model, and classes loaded.'.format(self.onnx_path))
        
    def get_model(self):
        print('Success to get model')
        return self.net
    
    def convert_to_onnx(self):
        image_data = np.array(np.random.randn(1, 3, self.input_shape[1], self.input_shape[0]),dtype='float32')

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            
            print('start export ONNX model')
            torch.onnx.export(self.net,               # 실행될 모델
                    images,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                    self.onnx_path,   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                    export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                    opset_version=17,          # 모델을 변환할 때 사용할 ONNX 버전
                    do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                    input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                    output_names = ['output'], # 모델의 출력값을 가리키는 이름
                    )
            print('finish export ONNX model')

    def detect_image(self, image, crop = False, count = False):
        image_shape = np.array(np.shape(image)[0:2])
       
        image       = cvtColor(image)

        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)

        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            # images = torch.from_numpy(image_data)
            input_name = self.net.get_inputs()[0].name

            outputs = self.net.run(None, {input_name: image_data})

            # decode
            outputs = [torch.from_numpy(output) for output in outputs]
            outputs = decode_outputs(outputs, self.input_shape)

            results = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return image

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textbbox((0, 0), label, font)[2:]#label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    