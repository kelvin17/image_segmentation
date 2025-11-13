import os
from PIL import Image

if __name__ == "__main__":
    name = "IMD020"
    DATA_PATH = '/dtu/datasets1/02516/PH2_Dataset_images/'
    image_path = os.path.join(DATA_PATH, f"{name}/{name}_Dermoscopic_Image/{name}.bmp")
    img = Image.open(image_path)
    print(f'{name}:{img.size}')l
    