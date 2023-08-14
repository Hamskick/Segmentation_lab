#
# #이미지 crop2_256
# import os
# from PIL import Image
#
# # 경로 설정
# img_folder = "C:\\Users\\admin\\Desktop\\Segmentation_lab\\1024_mask"
# output_folder = "C:\\Users\\admin\\Desktop\\Segmentation_lab\\try1_train_crop_mask_256"
#
# # 폴더 생성
# os.makedirs(output_folder, exist_ok=True)
#
# # 이미지 크기 설정
# original_size = (1024, 1024)
# crop_size = (256, 256)
#
# # 이미지 파일 목록 가져오기
# img_files = os.listdir(img_folder)
#
# # 파일 단위로 처리
# for idx, img_file in enumerate(img_files):
#     img_path = os.path.join(img_folder, img_file)
#
#     # 이미지 열기
#     img = Image.open(img_path)
#
#     # 이미지에서 16개의 이미지 추출
#     for i in range(16):
#         left = (i % 4) * crop_size[0]
#         top = (i // 4) * crop_size[1]
#         right = left + crop_size[0]
#         bottom = top + crop_size[1]
#
#         cropped_img = img.crop((left, top, right, bottom))
#
#         # 이미지 저장
#         output_index = idx * 16 + i
#         output_name = f"TRAIN_{str(output_index).zfill(6)}.png"
#         output_path = os.path.join(output_folder, output_name)
#         cropped_img.save(output_path)



# #보간법이용한 이미지 축소
# import os
# import cv2
#
# # 경로 설정
# img_folder = "C:\\Users\\admin\\Desktop\\Segmentation_lab\\try1_train_crop_mask_256"
# output_folder = "C:\\Users\\admin\\Desktop\\Segmentation_lab\\try1_train_crop_mask_256_2_224"
#
# # 폴더 생성
# os.makedirs(output_folder, exist_ok=True)
#
# # 이미지 크기 설정
# original_size = (256, 256)
# target_size = (224, 224)
#
# # 이미지 파일 목록 가져오기
# img_files = os.listdir(img_folder)
#
# # 파일 단위로 처리
# for idx, img_file in enumerate(img_files):
#     img_path = os.path.join(img_folder, img_file)
#     output_prefix = f"TRAIN_{str(idx)}"
#
#     # 이미지 읽기
#     img = cv2.imread(img_path)
#
#     # 이미지 축소
#     resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
#
#     # 이미지 저장
#     output_path = os.path.join(output_folder, f"{output_prefix}.png")
#     cv2.imwrite(output_path, resized_img)







# from PIL import Image
#
# # 이미지 파일 경로
# image_path = "C:\\Users\\admin\\Desktop\\Segmentation_lab\\1024_mask\\TRAIN_0013.png"
#
# # 이미지 열기
# image = Image.open(image_path)
#
# # 이미지 크기와 픽셀 형식 출력
# print("이미지 크기:", image.size)
# print("이미지 모드:", image.mode)
#
# # 픽셀 값 출력
# pixels = list(image.getdata())
# print("픽셀 값:")
# for pixel in pixels:
#     print(pixel)



import os
from PIL import Image
import random

satellite_image_folder = 'C:\\Users\\admin\\Desktop\\Segmentation_lab\\train_img'
mask_image_folder = 'C:\\Users\\admin\\Desktop\\Segmentation_lab\\1024_mask'
crop_image_folder = 'C:\\Users\\admin\\Desktop\\Segmentation_lab\\radom_train_crop'
crop_mask_folder = 'C:\\Users\\admin\\Desktop\\Segmentation_lab\\radom_train_mask_crop'
crop_size = 224  # Crop된 이미지의 크기
num_crops = 20  # 각 이미지당 생성할 crop 개수

satellite_image_files = os.listdir(satellite_image_folder)
mask_image_files = os.listdir(mask_image_folder)

# 이미지 파일들을 하나씩 불러와서 crop 작업 수행
for satellite_filename, mask_filename in zip(satellite_image_files, mask_image_files):
    satellite_image_path = os.path.join(satellite_image_folder, satellite_filename)
    mask_image_path = os.path.join(mask_image_folder, mask_filename)

    satellite_image = Image.open(satellite_image_path)
    mask_image = Image.open(mask_image_path)

    # Crop 작업을 수행하여 이미지를 6개씩 생성합니다.
    for i in range(num_crops):
        # 랜덤하게 crop 위치를 지정합니다.
        left = random.randint(0, satellite_image.width - crop_size)
        top = random.randint(0, satellite_image.height - crop_size)
        right = left + crop_size
        bottom = top + crop_size

        # 이미지와 마스크를 crop합니다.
        cropped_satellite_image = satellite_image.crop((left, top, right, bottom))
        cropped_mask_image = mask_image.crop((left, top, right, bottom))

        # Crop된 이미지와 마스크를 저장합니다.
        cropped_satellite_image_path = os.path.join(crop_image_folder, f'cropped_{i}_{satellite_filename}')
        cropped_mask_image_path = os.path.join(crop_mask_folder, f'cropped_{i}_{mask_filename}')
        os.makedirs(crop_image_folder, exist_ok=True)  # crop_image_folder 폴더가 없는 경우 생성
        os.makedirs(crop_mask_folder, exist_ok=True)  # crop_mask_folder 폴더가 없는 경우 생성
        cropped_satellite_image.save(cropped_satellite_image_path)
        cropped_mask_image.save(cropped_mask_image_path)

