import cv2
import sys
import os
import random
import numpy as np
from glob import glob
from datetime import datetime

import xml.etree.ElementTree as ET
import re
class DataAugmentation:
    def __init__(self):
        pass

    def scale(self, image, fx, fy):
        return cv2.resize(image, (0, 0), fx=fx, fy=fy)

    def rotate(self, image, angle):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        return cv2.warpAffine(image, M, (nW, nH)), M

    def flip(self, image, flip_code):
        return cv2.flip(image, flip_code)

    def adjust_brightness(self, image, value):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v = np.clip(v, 0, 255)
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    def histogram_equalization(self, image):
        if len(image.shape) == 3 and image.shape[2] == 3:
            img_y_cr_cb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            channels = cv2.split(img_y_cr_cb)
            cv2.equalizeHist(channels[0], channels[0])
            img_y_cr_cb = cv2.merge(channels)
            result_image = cv2.cvtColor(img_y_cr_cb, cv2.COLOR_YCrCb2BGR)
        else:
            result_image = cv2.equalizeHist(image)

        return result_image

    def adjust_gamma(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def adjust_saturation(self, image, saturation_factor=1.0):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    def translate_image(self, image, tx, ty):
        rows, cols = image.shape[:2]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_image = cv2.warpAffine(image, M, (cols, rows))

        return translated_image

    def save_image(self, image, path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # 使用 cv2.imencode 代替 cv2.imwrite 处理中文路径
        ext = os.path.splitext(path)[1]
        success, encoded_image = cv2.imencode(ext, image)
        if success:
            with open(path, 'wb') as f:
                f.write(encoded_image)

    def load_image_and_labels(self, image_path, label_path):
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)  # 使用 imdecode 读取含中文路径的图像
        labels = []

        tree = ET.parse(label_path)
        root = tree.getroot()

        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            labels.append([x1, y1, x2, y2])

        return image, labels

    def save_labels(self, label_path, labels, original_label_path, new_image):
        tree = ET.parse(original_label_path)
        root = tree.getroot()

        # 更新size节点中的width和height
        size_node = root.find('size')
        if size_node is not None:
            width_node = size_node.find('width')
            height_node = size_node.find('height')
            if width_node is not None:
                width_node.text = str(new_image.shape[1])
            if height_node is not None:
                height_node.text = str(new_image.shape[0])

        # 清除现有的object标签
        for obj in root.findall('object'):
            root.remove(obj)

        for label in labels:
            x1, y1, x2, y2 = label
            obj = ET.Element('object')
            original_obj = next((o for o in ET.parse(original_label_path).getroot().findall('object')
                                 if o.find('name').text), None)
            if original_obj is not None:
                for child in list(original_obj):
                    obj.append(child)

            bndbox = obj.find('bndbox')
            bndbox.find('xmin').text = str(int(x1))
            bndbox.find('ymin').text = str(int(y1))
            bndbox.find('xmax').text = str(int(x2))
            bndbox.find('ymax').text = str(int(y2))

            root.append(obj)

        tree.write(label_path)

    def process_image(self, image_path, label_path, fx, fy, angle_str, flip_code, brightness, he_flag, gamma,
                      saturation, tx_str, ty_str):
        if label_path is not None:
            image, labels = self.load_image_and_labels(image_path, label_path)
        else:
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)  # 使用 imdecode 读取含中文路径的图像
            labels = []  # 如果没有标注文件，则标签为空列表

        if image is not None:
            try:
                if '~' in angle_str:
                    match = re.match(r'(-?\d+(\.\d+)?)~(-?\d+(\.\d+)?)', angle_str)
                    if match:
                        angle_min, angle_max = map(float, match.groups()[::2])
                        angle = random.uniform(angle_min, angle_max)
                    else:
                        angle = float(angle_str or 0)  # 默认值0
                else:
                    angle = float(angle_str or 0)  # 默认值0
            except ValueError:
                angle = 0  # 当无法转换为浮点数时使用默认值0

            rotated_image, M = self.rotate(image, angle)
            if flip_code != 11:
                flipped_image = self.flip(rotated_image, flip_code)
            else:
                flipped_image = rotated_image
            bright_image = self.adjust_brightness(flipped_image, brightness)
            if he_flag == 1:
                he_image = self.histogram_equalization(bright_image)
            else:
                he_image = bright_image
            gamma_image = self.adjust_gamma(he_image, gamma)
            saturation_image = self.adjust_saturation(gamma_image, saturation)
            try:
                if '~' in tx_str:
                    match = re.match(r'(-?\d+(\.\d+)?)~(-?\d+(\.\d+)?)', tx_str)
                    if match:
                        tx_min, tx_max = map(float, match.groups()[::2])
                        tx = random.uniform(tx_min, tx_max)
                    else:
                        tx = float(tx_str or 0)  # 默认值0
                else:
                    tx = float(tx_str or 0)  # 默认值0
            except ValueError:
                tx = 0  # 当无法转换为浮点数时使用默认值0

            try:
                if '~' in ty_str:
                    match = re.match(r'(-?\d+(\.\d+)?)~(-?\d+(\.\d+)?)', ty_str)
                    if match:
                        ty_min, ty_max = map(float, match.groups()[::2])
                        ty = random.uniform(ty_min, ty_max)
                    else:
                        ty = float(ty_str or 0)  # 默认值0
                else:
                    ty = float(ty_str or 0)  # 默认值0
            except ValueError:
                ty = 0  # 当无法转换为浮点数时使用默认值0
            translate_image = self.translate_image(saturation_image, tx, ty)
            scaled_image = self.scale(translate_image, fx, fy)

            # 更新标签
            if labels:
                augmented_labels = self.update_labels(labels, scaled_image, M, flip_code, tx, ty, fx, fy, angle)
            else:
                augmented_labels = []

            return scaled_image, augmented_labels
        else:
            print(f"Error loading image {image_path}")
            return None, None

    def update_labels(self, labels, image, M, flip_code, tx, ty, fx, fy, angle):
        augmented_labels = []

        # Step 1: Apply affine transformation (rotation and translation)
        for label in labels:
            x1, y1, x2, y2 = label
            points = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
            points = np.hstack([points, np.ones((4, 1))])
            new_points = M.dot(points.T).T
            x_coords = new_points[:, 0]
            y_coords = new_points[:, 1]
            new_x1, new_y1 = x_coords.min(), y_coords.min()
            new_x2, new_y2 = x_coords.max(), y_coords.max()
            augmented_labels.append([new_x1, new_y1, new_x2, new_y2])
            # print(new_x1, new_y1, new_x2, new_y2)

        # Step 2: Apply flipping
        h, w, _ = image.shape
        h=h/fy
        w=w/fx
        flip_labels = []
        for label in augmented_labels:
            x_min, y_min, x_max, y_max = label
            if flip_code == 0:
                flip_labels.append([x_max, h - y_min, x_min, h - y_max])
            elif flip_code == 1:
                flip_labels.append([w - x_max, y_min, w - x_min, y_max])
            elif flip_code == -1:
                flip_labels.append([w - x_min, h - y_max, w - x_max, h - y_min])
            elif flip_code == 11:
                flip_labels = augmented_labels

        # Step 3: Apply translation
        translated_labels = []
        for label in flip_labels:
            new_x1, new_y1, new_x2, new_y2 = label
            # print(new_x1, new_y1, new_x2, new_y2)
            translated_labels.append([new_x1 + tx, new_y1 + ty, new_x2 + tx, new_y2 + ty])
            # print(new_x1 + tx, new_y1 + ty, new_x2 + tx, new_y2 + ty)

        # Step 4: Apply scaling
        scaled_labels = []
        for label in translated_labels:
            x_min, y_min, x_max, y_max = label
            scaled_labels.append([x_min * fx, y_min * fy, x_max * fx, y_max * fy])
            # print(x_min * fx, y_min * fy, x_max * fx, y_max * fy)

        return scaled_labels