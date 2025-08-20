import cv2
import numpy as np

def overlay_images(rgb_image_path, depth_image_path, output_path, alpha=0.7):
    """
    RGB와 Depth 이미지를 오버레이하고 결과 이미지를 파일로 저장합니다.

    Args:
        rgb_image_path (str): RGB 이미지 파일 경로.
        depth_image_path (str): Depth 이미지 파일 경로.
        output_path (str): 오버레이된 이미지 저장 경로.
        alpha (float): RGB와 Depth의 가중치. 0에서 1 사이의 값. 기본값은 0.5.
    """
    # RGB 이미지와 Depth 이미지 불러오기
    rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)

    # Depth 이미지를 컬러화 (컬러 맵 적용)
    depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=255.0/np.max(depth_image)), cv2.COLORMAP_JET)

    # RGB와 Depth 컬러화된 이미지를 오버레이
    overlay_image = cv2.addWeighted(rgb_image, alpha, depth_colored, 1 - alpha, 0)

    # 결과 저장
    cv2.imwrite(output_path, overlay_image)
    print(f"오버레이된 이미지를 '{output_path}'에 저장했습니다.")

# 예제 사용법
overlay_images("/home/rise/Desktop/graduation/CalibNet/datasets/CUSTOM/images/RGB_rec/_2023-06-21-20-14-18_frame000002.png", "/home/rise/Desktop/graduation/CalibNet/datasets/CUSTOM/images/depth_ensenso_mm/_2023-06-21-20-14-18_frame000002.png", "/home/rise/overlay_result_rec.png")
