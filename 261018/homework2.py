import cv2
import matplotlib.pyplot as plt
import numpy as np

basketball = cv2.imread('src/images/basketball_crop.jpg')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

lower_basketball_lab = np.array([70, 130, 70])
upper_basketball_lab = np.array([180, 230, 185])

basketball = cv2.imread('src/images/basketball_crop.jpg')

# TODO: LAB 색공간으로 변환
basketball_lab = cv2.cvtColor(basketball, cv2.COLOR_BGR2LAB)

# TODO: 각 채널 분리
l, a, b = cv2.split(basketball_lab)

# TODO: CLAHE 생성 (clipLimit은 3.0, tileGridSize는 8x8로 설정)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# TODO: L 채널에 CLAHE 적용 후 강화된 이미지 생성
l_clahe_enhanced = clahe.apply(l) 
basketball_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))

# TODO: 강화된 이미지로 binary mask 생성
mask_enhanced = cv2.inRange(basketball_enhanced_lab, lower_basketball_lab, upper_basketball_lab)

ball_mask_opened = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel)

# TODO: Morphological operation을 적용하여 최종 binary mask 생성
final_ball_mask = cv2.morphologyEx(ball_mask_opened, cv2.MORPH_CLOSE, kernel)

final_ball_img = basketball.copy()

# TODO: Binary mask로 외곽선 검출
contours, _ = cv2.findContours(
    final_ball_mask,        # binary mask
    cv2.RETR_EXTERNAL,       # 구멍 뚫린 객체의 바깥쪽 테두리만 검출
    cv2.CHAIN_APPROX_SIMPLE  # 직선 경로 위에 있는 픽셀 좌표 전체 대신 꼭짓점 좌표만 저장
)

# TODO: 최종 농구공 영역 및 외곽선 선택
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

plt.imshow(final_img)
plt.axis("off");