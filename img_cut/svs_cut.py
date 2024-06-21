# 注意：这是帅逼张家浙的代码#
import os
os.add_dll_directory('D:/Anaconda/envs/py39/Lib/openslide-bin-4.0.0.3-windows-x64/bin')
from cv2 import cv2
import openslide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np



def split_image(src_path):
    img = openslide.open_slide(src_path)
    save_path = 'E:/python/data/my data/img_cut/1/' #保存路径
    [w,h] = img.level_dimensions[0]
    print(w,h)
    data_gen = DeepZoomGenerator(img, tile_size=256, overlap=0, limit_bounds=False)

    height = 256
    width = 256
    num_w = int(np.floor(w/width))+1
    num_h = int(np.floor(h/height))+1

    for i in range(num_w):
        for j in range(num_h):
            child_img = np.array(data_gen.get_tile(16, (i, j)))
            #io.imsave(join(save_path, "02" + str(i) + '_' + str(j) + ".png"), img)  # 保存图像
            cv2.imwrite(save_path + str(i) + str(j) + '.png', child_img)




if __name__ == '__main__':
    # src_path 具体图片路径，包含后缀
    src_path = 'E:/python/data/my data/SVS/data/1027476.svs' #svs路径
    split_image(src_path)