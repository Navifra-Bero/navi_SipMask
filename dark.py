import cv2
import os

def darken_images_in_folder(input_folder, output_folder, darken_factors=(0.05, 0.4, 1.0)):
    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 입력 폴더의 이미지 파일 리스트 가져오기
    file_list = sorted(os.listdir(input_folder))  # 파일을 정렬하여 순서 유지
    total_files = len(file_list)

    # 각 구간의 경계 계산
    first_third = total_files // 3
    second_third = 2 * total_files // 3

    # 파일 처리
    for i, filename in enumerate(file_list):
        input_path = os.path.join(input_folder, filename)

        # 이미지 파일인지 확인
        if os.path.isfile(input_path):
            # 이미지 읽기
            image = cv2.imread(input_path)

            if image is not None:
                # 해당 이미지에 적용할 darken_factor 결정
                if i < first_third:
                    darken_factor = darken_factors[0]
                elif i < second_third:
                    darken_factor = darken_factors[1]
                else:
                    darken_factor = darken_factors[2]

                # 이미지 어둡게 만들기
                darkened_image = (image * darken_factor).astype('uint8')

                # 출력 경로 설정
                output_path = os.path.join(output_folder, filename)
                
                # 이미지 저장
                cv2.imwrite(output_path, darkened_image)
                print(f"{i + 1}/{total_files} - {filename} 처리 완료 (darken_factor={darken_factor})")

    print(f"모든 이미지가 {output_folder}에 저장되었습니다.")

# 사용 예시
input_folder = '/home/rise/Documents/pkb/images/RGB'  # 입력 이미지 폴더 경로
output_folder = '/home/rise/Documents/pkb/images/RGB_dark_2'  # 출력 이미지 폴더 경로
darken_images_in_folder(input_folder, output_folder)
