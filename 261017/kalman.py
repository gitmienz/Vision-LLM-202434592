import cv2
import numpy as np
from numpy.linalg import inv  # inv 함수 임포트 추가

def KalmanFilter(mu_prev, sigma_prev, z):
    mu_bar = A_t.dot(mu_prev)
    sigma_bar = A_t.dot(sigma_prev).dot(A_t.transpose()) + R_t
    if z is None:
        return mu_bar, sigma_bar
    else:
        K_t = sigma_bar.dot(C_t.transpose()).dot(inv(C_t.dot(sigma_bar).dot(C_t.transpose()) + Q_t))
        mu = mu_bar + K_t.dot(z - C_t.dot(mu_bar))
        sigma = (np.identity(2) - K_t.dot(C_t)).dot(sigma_bar)
        return mu, sigma

# 칼만 필터 변수 정의
A_t = np.array([[1, 1], [0, 1]])
G = np.array([[0.5], [1]])
R_t = G.dot(G.transpose())
C_t = np.array([[1, 0]])
Q_t = np.array([[1]])

# 2x1 열벡터 및 2x2 공분산 행렬로 수정
mu_t = np.array([[0.0], [0.0]])
sigma_t = np.eye(2)

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 상태 변수 초기화 (루프 바깥)
found = False
radius = 0
x_bel, y_bel = 0.0, 0.0

green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    ball_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(ball_lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    l_clahe_enhanced = clahe.apply(l) 
    ball_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))
    mask_enhanced = cv2.inRange(ball_enhanced_lab, green_lower, green_upper)
    ball_mask_opened = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel)
    final_ball_mask = cv2.morphologyEx(ball_mask_opened, cv2.MORPH_CLOSE, kernel)

    contour_lst, _ = cv2.findContours(
        final_ball_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 면적 기준 가장 큰 외곽선 추출
    if len(contour_lst) > 0:
        contour = max(contour_lst, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))

        # 검출된 객체 시각화 (빨간 원)
        cv2.circle(frame, center, int(radius), (0, 0, 255), 2)
        
        # 최초 검출 시 칼만 필터 위치 초기화
        if not found:
            found = True
            mu_t[0][0] = center[0]  # x 좌표로 초기화

        # 측정값 z: x 좌표 (1x1 행렬)
        z = np.array([[float(center[0])]])
        mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, z)
        x_bel = mu_t[0][0]
        y_bel = center[1]  # y축은 측정이 안 되므로 현재 관측값 사용
    elif found:
        # 가림(Occlusion) 상태 -> 예측만 진행
        mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, None)
        x_bel = mu_t[0][0]

    # 예측한 객체 위치 오버레이 (노란 원)
    if found:
        cv2.circle(frame, (int(x_bel), int(y_bel)), int(radius), (0, 255, 255), 2)

    cv2.imshow("VideoCapture with GStreamer", frame)

cap.release()
cv2.destroyAllWindows()