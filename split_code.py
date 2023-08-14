# import os
# import shutil
# from torchvision.datasets import ImageFolder
#
# # 데이터셋 경로 설정
# dataset_path = 'C:\\Users\\admin\\Desktop\\Segmentation_lab\\try1_train_crop_256'
#
# # 분할된 데이터셋 저장 경로 설정
# train_path = 'C:\\Users\\admin\\Desktop\\crop_256\\train'
# val_path = 'C:\\Users\\admin\\Desktop\\crop_256\\val'
#
# # 분할 비율 설정
# train_ratio = 0.8
#
# # 분할 비율에 따라 이미지를 분할하는 함수
# def split_dataset(dataset_path, train_path, val_path, train_ratio):
#     dataset = ImageFolder(dataset_path)
#     classes = dataset.classes
#     num_images = len(dataset)
#     num_train = int(train_ratio * num_images)
#     num_val = num_images - num_train
#
#     # 클래스별로 분할된 데이터셋 폴더 생성
#     os.makedirs(train_path, exist_ok=True)
#     os.makedirs(val_path, exist_ok=True)
#     for class_name in classes:
#         os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
#         os.makedirs(os.path.join(val_path, class_name), exist_ok=True)
#
#     # 이미지를 순서대로 분할하여 저장
#     for idx, (image_path, label) in enumerate(dataset.samples):
#         class_name = classes[label]
#         if idx < num_train:
#             shutil.copy(image_path, os.path.join(train_path, class_name))
#         else:
#             shutil.copy(image_path, os.path.join(val_path, class_name))
#
#     print(f"Dataset split complete. Train: {num_train}, Validation: {num_val}")
#
# # 이미지 분할 실행
# split_dataset(dataset_path, train_path, val_path, train_ratio)


#8:2 분할 코드
import os
import shutil

# 이미지 폴더 경로 설정
image_folder_path = 'C:\\Users\\admin\\Desktop\\remove_dot_data\\remove_dot_mask'

# 분할된 데이터셋 저장 경로 설정
train_path = 'C:\\Users\\admin\\Desktop\\real_datasets\\train_labels'
val_path = 'C:\\Users\\admin\\Desktop\\real_datasets\\val_labels'

# 분할 비율 설정
train_ratio = 0.8

# 이미지 폴더 내 이미지를 순서대로 분할하는 함수
def split_images(image_folder_path, train_path, val_path, train_ratio):
    # 이미지 파일 목록 읽기
    image_files = sorted([f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f)) and f.lower().endswith('.png')])

    # 분할된 데이터셋 폴더 생성
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    # 이미지 파일을 순서대로 분할하여 저장
    num_images = len(image_files)
    num_train = int(train_ratio * num_images)
    num_val = num_images - num_train

    # 훈련 데이터 복사
    for idx in range(num_train):
        image_file = image_files[idx]
        shutil.copy(os.path.join(image_folder_path, image_file), train_path)

    # 검증 데이터 복사
    for idx in range(num_train, num_images):
        image_file = image_files[idx]
        shutil.copy(os.path.join(image_folder_path, image_file), val_path)

    print(f"Image split complete. Train: {num_train}, Validation: {num_val}")

# 이미지 분할 실행
split_images(image_folder_path, train_path, val_path, train_ratio)

