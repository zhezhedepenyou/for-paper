# 注意：这是帅逼张家浙的代码#
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
from cv2 import cv2


def split_image(src_path):
    img = cv2.imread(src_path)
    # cv2.imwrite(path, img)
    size = img.shape[0:2]
    w = size[1]
    h = size[0]
    # print(file, w, h)
    # 每行的高度和每列的宽度
    row_height = 256
    col_width = 256
    rownum=h//256-1
    colnum=w//256-1
    num = 0

    for i in range(rownum):
        for j in range(colnum):
            # 保存切割好的图片的路径，记得要填上后缀，以及名字要处理一下，可以是
            # src_path.split('.')[0] + '_' + str((i+1)*(j+1)) + '.jpg'
            save_path = 'E:/python/data/my data/img_cut/1/'
            row_start = j * col_width
            row_end = (j + 1) * col_width
            col_start = i * row_height
            col_end = (i + 1) * row_height
            # print(row_start, row_end, col_start, col_end)
            # cv2图片： [高， 宽]
            #child_img = img[256, 256]
            child_img = img[col_start:col_end, row_start:row_end]
            cv2.imwrite(save_path+str(i)+str(j)+'.png', child_img)


if __name__ == '__main__':
    # 可以遍历文件夹
    # file_path = r'我是路径（文件夹路径）'
    # for file in file_names:

    # src_path 具体图片路径，包含后缀
    src_path = 'E:/python/data/my data/SVS/data/1027307.png'
    row = 4
    col = 4
    #split_image(src_path, row, col, file.split('.')[0])
    split_image(src_path)