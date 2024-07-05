import cv2
import numpy as np
import os
import shutil
from pynput.mouse import Listener
import dlib
from imutils import face_utils
import imutils

root = "./eye_images/"  # 使用相对路径
padding = 10  # 眼睛区域扩展的像素数

if not os.path.isdir(root):
    os.mkdir(root)


def normalize(x):
    minn, maxx = x.min(), x.max()
    return (x - minn) / (maxx - minn)


def scan(image_size=(224, 224)):  # 调整图像大小以更好地匹配模型需求
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)  # 使用dlib检测面部

    if len(rects) == 1:  # 确保检测到一个面部
        rect = rects[0]
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # 提取左眼和右眼的坐标
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # 计算扩展后的眼睛区域
        leftEyeBounds = [max(min(leftEye[:, 0]) - padding, 0),
                         min(max(leftEye[:, 0]) + padding, gray.shape[1]),
                         max(min(leftEye[:, 1]) - padding, 0),
                         min(max(leftEye[:, 1]) + padding, gray.shape[0])]
        rightEyeBounds = [max(min(rightEye[:, 0]) - padding, 0),
                          min(max(rightEye[:, 0]) + padding, gray.shape[1]),
                          max(min(rightEye[:, 1]) - padding, 0),
                          min(max(rightEye[:, 1]) + padding, gray.shape[0])]

        # 从面部坐标提取眼部区域
        leftEyeImage = gray[leftEyeBounds[2]:leftEyeBounds[3], leftEyeBounds[0]:leftEyeBounds[1]]
        rightEyeImage = gray[rightEyeBounds[2]:rightEyeBounds[3], rightEyeBounds[0]:rightEyeBounds[1]]
        leftEyeImage = cv2.resize(leftEyeImage, (224, 224))
        rightEyeImage = cv2.resize(rightEyeImage, (224, 224))
        leftEyeImage = normalize(leftEyeImage)
        rightEyeImage = normalize(rightEyeImage)
        if leftEyeImage is not None and rightEyeImage is not None:
            eyes = np.hstack([leftEyeImage, rightEyeImage])
            return (eyes * 255).astype(np.uint8)
    else:
        print("Face detected:", len(rects))
        return None


def extract_eye(frame, eye_points, image_size, padding=15):
    # 根据眼部坐标生成边界框，同时添加padding
    x, y, w, h = cv2.boundingRect(np.array([eye_points]))
    # 通过padding增加边界，同时确保边界不超过图像边缘
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, frame.shape[1] - x)
    h = min(h + 2 * padding, frame.shape[0] - y)
    eye = frame[y:y + h, x:x + w]
    eye = cv2.resize(eye, image_size)  # 根据提供的尺寸调整图像
    eye = normalize(eye)
    return eye


def on_click(x, y, button, pressed):
    if pressed:
        eyes = scan()
        if eyes is not None:
            filename = f"{root}{x}_{y}_{button}.jpeg"
            cv2.imwrite(filename, eyes)
            print(f"Saved: {filename}")
        else:
            print("No eye image to save.")


# 初始化dlib的面部检测器和面部标记预测器
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"  # 这里填写预测器文件路径
predictor = dlib.shape_predictor(predictor_path)

video_capture = cv2.VideoCapture(0)

with Listener(on_click=on_click) as listener:
    listener.join()
