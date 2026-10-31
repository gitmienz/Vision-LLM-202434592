import time
import cv2
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt


IMAGE_PATH = "../src/images/seagull.jpg"
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# 이미지 불러오기 및 기본 처리
img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

img_gray = cv2.resize(
    img_gray,
    (TARGET_WIDTH, TARGET_HEIGHT),
    interpolation=cv2.INTER_AREA
)

input_np = img_gray.astype(np.float32) / 255.0

# x축 방향 edge (세로 선) 검출 필터
dx_edge_filter = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ],
    dtype=np.float32
)

# y축 방향 edge (가로 선) 검출 필터
dy_edge_filter = np.array(
    [
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ],
    dtype=np.float32
)

# 너무 약한 특징 (0.2 이하)은 무시
dx_bias = -0.2
dy_bias = -0.2

def relu(x):
    return np.maximum(0, x)

# CPU convolution2d
def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    padded_image = np.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )

    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    output = np.zeros((output_height, output_width), dtype=np.float32)

    for output_y in range(output_height):
        for output_x in range(output_width):
            start_y = output_y * stride
            start_x = output_x * stride

            image_region = padded_image[
                start_y:start_y + kernel_height,
                start_x:start_x + kernel_width
            ]

            weighted_sum = np.sum(image_region * kernel)
            output[output_y, output_x] = weighted_sum + bias

    return output


# CuPy convolution2d 
def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
    kernel_height, kernel_width = kernel.shape

    padded_image = cp.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )

    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1
    output = cp.full((output_height, output_width), bias, dtype=cp.float32)

    for i in range(kernel_height):
        for j in range(kernel_width):
            weight = kernel[i, j]
            
            if weight != 0:
                input_region = padded_image[
                    i : i + output_height * stride : stride,
                    j : j + output_width * stride : stride
                ]
                output += input_region * weight

    return output


# CPU 처리
cpu_start = time.perf_counter()

output_dx_np = convolution2d(
    image=input_np,
    kernel=dx_edge_filter,
    bias=dx_bias,
    stride=1,
    padding=1
)
output_dy_np = convolution2d(
    image=input_np,
    kernel=dy_edge_filter,
    bias=dy_bias,
    stride=1,
    padding=1
)

cpu_end = time.perf_counter()
cpu_time = cpu_end - cpu_start

# 변수명 오류 수정 (vertical_output -> output_dx_np)
relu_dx_np = relu(output_dx_np)
relu_dy_np = relu(output_dy_np)


# GPU 처리 (변환-워밍업-처리 파이프라인)
#  변환
input_cp = cp.asarray(input_np)
dx_edge_filter_cp = cp.asarray(dx_edge_filter)
dy_edge_filter_cp = cp.asarray(dy_edge_filter)

#  워밍업
warmup_image = cp.zeros((64, 64), dtype=cp.float32)
warmup_kernel = cp.zeros((3, 3), dtype=cp.float32)
_ = convolution2d_cp(warmup_image, warmup_kernel, bias=0.0, stride=1, padding=1)
cp.cuda.Stream.null.synchronize()

#  GPU 연산
cp.cuda.Stream.null.synchronize()
gpu_start = time.perf_counter()

output_dx_cp = convolution2d_cp(
    image=input_cp,
    kernel=dx_edge_filter_cp,
    bias=dx_bias,
    stride=1,
    padding=1
)
output_dy_cp = convolution2d_cp(
    image=input_cp,
    kernel=dy_edge_filter_cp,
    bias=dy_bias,
    stride=1,
    padding=1
)

cp.cuda.Stream.null.synchronize()
gpu_end = time.perf_counter()
gpu_time = gpu_end - gpu_start


output_dx_cp_np = cp.asnumpy(output_dx_cp)
output_dy_cp_np = cp.asnumpy(output_dy_cp)

relu_dx_cp_np = relu(output_dx_cp_np)
relu_dy_cp_np = relu(output_dy_cp_np)


# 결과 및 비교 출력
print("========== NumPy CPU ==========")
print(f"Processing time: {cpu_time:.4f} seconds")
print()

print("========== CuPy GPU ==========")
print(f"GPU processing: {gpu_time:.4f} seconds")
print(f"Speedup: {cpu_time / gpu_time:.2f}x")
print()


# 시각화
plt.figure(figsize=(18, 8))

plt.subplot(2, 3, 1), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
plt.subplot(2, 3, 2), plt.imshow(relu_dx_np, cmap="gray"), plt.title("Vertical Convolution (CPU)")
plt.subplot(2, 3, 3), plt.imshow(relu_dx_cp_np, cmap="gray"), plt.title("Vertical Convolution (GPU)")

plt.subplot(2, 3, 4), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
plt.subplot(2, 3, 5), plt.imshow(relu_dy_np, cmap="gray"), plt.title("Horizontal Convolution (CPU)")
plt.subplot(2, 3, 6), plt.imshow(relu_dy_cp_np, cmap="gray"), plt.title("Horizontal Convolution (GPU)")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
