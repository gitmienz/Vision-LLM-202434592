import numpy as np
import matplotlib.pyplot as plt
import skimage.data
from random import randint, gauss
import cv2

def draw_bbox(img, coord1, coord2, color, thickness=3, txt=""):
    _, txt_h = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(img, coord1, coord2, color, thickness)
    cv2.putText(img, txt, (coord1[0], coord1[1]-txt_h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# TODO: 1) BBox 좌표를 (x,y,w,h)에서 (x1,y1,x2,y2)로 변환하는 함수 정의
# TODO: 2) IoU 계산 함수 정의

def convert_bbox_coord(bbox):
    x, y, w, h = bbox
    x1, y1 = x, y
    x2, y2 = x+w, y+h
    return x1, y1, x2, y2

def calculate_iou(bbox_a, bbox_b):

    xa_1, ya_1, xa_2, ya_2 = bbox_a
    xb_1, yb_1, xb_2, yb_2 = bbox_b
    
    intersection_x1 = max(xa_1, xb_1)
    intersection_y1 = max(ya_1, yb_1)
    intersection_x2 = min(xa_2, xb_2)
    intersection_y2 = min(ya_2, yb_2)
    
    if intersection_x2-intersection_x1 > 0 and intersection_y2-intersection_y1 > 0:
        intersection_area = (intersection_x2-intersection_x1) * (intersection_y2 - intersection_y1)
        union_area = abs(xa_1-xa_2)*abs(ya_1-ya_2)+abs(xb_1-xb_2)*abs(yb_1-yb_2) - intersection_area
    else: 
        return 0

    # TODO: ZeroDivisionError 주의
    return intersection_area / union_area;

# 랜덤으로 박스생성
def random_bbox(img, min_w=100, min_h=400, max_w=600, max_h=600):
    img_h, img_w = img.shape[:2]

    w = randint(min_w, min(max_w, img_w))
    h = randint(min_h, min(max_h, img_h))

    x1 = randint(0, img_w - w)
    y1 = randint(0, img_h - h)

    return x1, y1, x1 + w, y1 + h

def random_conf():
    return min(1.0, max(0.0, gauss(0.6, 0.1)))


hubble = skimage.data.hubble_deep_field()

N = 20

boxes = []
scores = []

for i in range(N):
    boxes.append(random_bbox(hubble))
    scores.append(random_conf())

for i, box in enumerate(boxes):
    draw_bbox(
        hubble,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{scores[i]:.2f}"
    )

plt.figure(figsize=(10,8))
plt.imshow(hubble)
plt.axis("off");


# TODO: Confidence Thresholding 함수 정의
# TODO: Confidence Score가 특정 임계값보다 낮으면 해당 BBox 제거

def prune_low_conf(boxes, scores, conf_threshold=0.5):
    """
    Confidence Thresholding 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    conf_threshold : float
        Bounding Box를 유지하기 위한 최소 Confidence Score

    Returns
    -------
    pruned_boxes : list
        Confidence Score가 임계값 이상인 Bounding Box 목록

    pruned_scores : list
        Confidence Score가 임계값 이상인 Bounding Box의 Confidence Score 목록
    """
    pruened_boxes = []
    pruened_scores = []

    for box, score in zip(boxes, scores):
        if score > conf_threshold:
            pruened_boxes.append(box)
            pruened_scores.append(score)
    return pruened_boxes, pruened_scores



# TODO: NMS 함수 정의
# TODO: np.sort() 함수를 사용하면 1차원 배열을 정렬
# TODO: np.argsort() 함수를 사용하면 1차원 배열의 정렬된 인덱스를 반환

def apply_nms(boxes, scores, iou_threshold=0.5):
    
    """
    NMS 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    iou_threshold : float
        중복 Bounding Box를 제거할 IoU 기준값

    Returns
    -------
    nms_boxes : list
        NMS 적용 후 유지된 Bounding Box 목록

    nms_scores : list
        NMS 적용 후 유지된 Bounding Box의 Confidence Score 목록
    """
    
    nms_boxes = []
    nms_scores = []
    
    # 박스와 점수 튜플형태로 정렬
    box_score = zip(scores, boxes)
    sort_box = sorted(box_score, key=lambda x:x[0], reverse=True)
    
    while sort_box:
        score, box = sort_box.pop(0)
        nms_boxes.append(box)
        nms_scores.append(score)
        clear = []
        
        for score2, box2 in sort_box:
            if calculate_iou(box, box2) <= iou_threshold:
                clear.append((score2, box2))
        
        sort_box = clear   
                
    return nms_boxes, nms_scores

# confidence thresholding 과  NMS 적용
print(len(boxes), len(scores))

new_boxes, new_scores = prune_low_conf(boxes, scores)
new_boxes, new_scores = apply_nms(boxes, scores, iou_threshold=0.2)

print(len(new_boxes), len(new_scores))


hubble = skimage.data.hubble_deep_field()

for i, box in enumerate(new_boxes):
    draw_bbox(
        hubble,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{scores[i]:.2f}"
    )

plt.figure(figsize=(10,8))
plt.imshow(hubble)
plt.axis("off");