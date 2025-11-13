import os
from PIL import Image
import numpy as np
import cv2

if __name__ == "__main__":
    name = "IMD020"
    DATA_PATH = '/dtu/datasets1/02516/PH2_Dataset_images/'
    # image_path = os.path.join(DATA_PATH, f"{name}/{name}_Dermoscopic_Image/{name}.bmp")
    # img = Image.open(image_path)
    # print(f'{name}:{img.size}')
    # print(img.getpixel((700,500)))
    
    print('---'* 20 + 'image read' + '---'* 20)
    image_path = os.path.join('/zhome/b7/2/219221/IDLCV_Exercise_3_segmentation/dataset/DRIVE/test/labels', '01_manual1.png')
    img = Image.open(image_path)
    print(f'size:{img.size}')
    print(img.getpixel((0,0)))
    
    img_np = np.array(img)
    print("dtype:", img_np.dtype)
    print("shape:", img_np.shape)
    print("min:", img_np.min())
    print("max:", img_np.max())
    print("unique:", np.unique(img_np))
    
    print('---'* 20 + 'cv2 read' + '---'* 20)
    img_path = '/zhome/b7/2/219221/IDLCV_Exercise_3_segmentation/dataset/DRIVE/test/labels/01_manual1.png'
    # img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    # img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    
    print("类型:", type(img))
    print("dtype:", img.dtype)
    print("shape:", img.shape)
    print("最小值:", img.min())
    print("最大值:", img.max())

    # 查看像素分布
    print("像素值唯一数量:", np.unique(img))
    # print("R通道 min/max:", img[:,:,0].min(), img[:,:,0].max())
    # print("G通道 min/max:", img[:,:,1].min(), img[:,:,1].max())
    # print("B通道 min/max:", img[:,:,2].min(), img[:,:,2].max())


    