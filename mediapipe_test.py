import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from scipy.interpolate import interp1d

# 初始化 MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 获取屏幕大小以便映射鼠标位置
screen_width, screen_height = pyautogui.size()

# 创建映射函数，增大输出范围
# 仅调整水平映射函数的输出范围
# 调整水平映射函数，增大输出范围并压缩输入范围
map_x = interp1d([0.1, 0.9], [-30 * screen_width, 30 * screen_width], fill_value="extrapolate")

# 垂直映射函数保持不变
map_y = interp1d([0, 1], [0, screen_height], fill_value="extrapolate")


# 平滑移动的缓存
smooth_x, smooth_y = 0, 0
alpha = 0.5  # 平滑因子

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # 转换颜色空间从 BGR 到 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 处理图像，检测面部
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 提取左眼和右眼的中心坐标
                left_eye_x = np.mean([face_landmarks.landmark[i].x for i in range(362, 382)])
                left_eye_y = np.mean([face_landmarks.landmark[i].y for i in range(362, 382)])
                right_eye_x = np.mean([face_landmarks.landmark[i].x for i in range(133, 153)])
                right_eye_y = np.mean([face_landmarks.landmark[i].y for i in range(133, 153)])

                # 计算两眼中心坐标
                eye_x = (left_eye_x + right_eye_x) / 2
                eye_y = (left_eye_y + right_eye_y) / 2

                # 映射到屏幕坐标并平滑处理
                target_x = map_x(eye_x)
                target_y = map_y(eye_y)
                smooth_x = alpha * smooth_x + (1 - alpha) * target_x
                smooth_y = alpha * smooth_y + (1 - alpha) * target_y

                # 移动鼠标
                pyautogui.moveTo(smooth_x, smooth_y)

        # 显示图像
        cv2.imshow('MediaPipe Eye Tracking', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
