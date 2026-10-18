import cv2
import numpy as np

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

while True:
    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # TODO: 배운 기술(이미지 전처리, 특징 추출)을 활용하여 실시간으로 객체를 검출

    # 초록 LAB lower/upper range
    green_lower = np.array([30, 60, 90], dtype=np.uint8)
    green_upper = np.array([230, 115, 180], dtype=np.uint8)

    ball_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    
    l, a, b = cv2.split(ball_lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    l_clahe_enhanced = clahe.apply(l) 
    ball_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))
    mask_enhanced = cv2.inRange(ball_enhanced_lab, green_lower, green_upper)
    ball_mask_opened = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel)

    final_ball_mask = cv2.morphologyEx(ball_mask_opened, cv2.MORPH_CLOSE, kernel)

    final_ball_img = frame.copy()

    contours, _ = cv2.findContours(
        final_ball_mask,        # binary mask
        cv2.RETR_EXTERNAL,       # 구멍 뚫린 객체의 바깥쪽 테두리만 검출
        cv2.CHAIN_APPROX_SIMPLE  # 직선 경로 위에 있는 픽셀 좌표 전체 대신 꼭짓점 좌표만 저장
    )

    if len(contours) > 0:
        # 가장 큰 contour
        largest_contour = max(contours, key=cv2.contourArea)

        # 가장 큰 contour만 채운 binary mask 생성
        final_mask = np.zeros_like(final_ball_mask)
        cv2.drawContours(
            final_mask,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

        # 가장 큰 contour만 채운 binary mask를 반투명 초록색으로 overlay
        mask_color = np.zeros_like(final_ball_img)
        mask_color[:, :, 1] = final_mask
        alpha = 0.35
        final_img = cv2.addWeighted(final_ball_img, 1.0, mask_color, alpha, 0)

        # Contour 그리기 (초록색 외곽선)
        cv2.drawContours(final_img, [largest_contour], -1, (0,255,0), 2)

        # 최소 외접원 중심 및 반지름
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center_circle = (int(x), int(y))
        radius = int(radius)

        # 파란색 최소 외접원
        cv2.circle(final_img, center_circle, radius, (0,0,255), 3)

        cv2.imshow("VideoCapture with GStreamer", final_img)



    

cap.release()
cv2.destroyAllWindows()