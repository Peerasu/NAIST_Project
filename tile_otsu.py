import os
import pandas as pd 
import csv
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


def do_thresholding(img):
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_c = 255 - grayscale_img
    thres, thres_img = cv2.threshold(img_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thres, thres_img, img_c


def do_image_cut(image, tile_size, name, save_path, annot_path, tile_annot_path, Cut=1):
    # image = cv2.resize(image, dsize=(resize, resize), interpolation=cv2.INTER_LINEAR)
    # print(image.shape)
    Thres_,_,_ = do_thresholding(image)
    tile_size = int(tile_size)
    Image_Thres = (Thres_ - 20) * tile_size * tile_size * 3
    # print(Image_Thres, Thres_)
    image_shape = image.shape
    h_num = int(math.ceil(image_shape[0] / tile_size))
    w_num = int(math.ceil(image_shape[1] / tile_size))
    # print(w_num, h_num)
    image = cv2.resize(image, dsize=(h_num*tile_size, w_num*tile_size), interpolation=cv2.INTER_LINEAR)
    # Block = np.zeros((tile_size, tile_size, image.shape[2]), dtype=np.uint8)
    
    annot_file = pd.read_csv(annot_path)
    for i, sample_name in enumerate(annot_file['Sample_Name']):
        if name == sample_name:
            date = (annot_file.iloc[i])['Date']
            death = (annot_file.iloc[i])['Death']
            break
        
    tile_list = []
    bg_list = []
    count = 0
    if Cut == 0:
        for i in range(h_num):
            for j in range(w_num):
                tile = image[i*tile_size:(i+1)*tile_size,j*tile_size:(j+1)*tile_size]
                background = 0
                label = [date, death, background]
                tile_write(tile, count, name, save_path, label, tile_annot_path)
                count += 1
                tile_list.append(tile)
                bg_list.append(background)
    else:
        for i in range(h_num):
            for j in range(w_num):
                tile = image[i*tile_size:(i+1)*tile_size,j*tile_size:(j+1)*tile_size]
                # otsu thresholding
                tile_iN = 255 - tile
                Tile_Thres = np.sum(tile_iN)
                
                if Tile_Thres > Image_Thres:
                    background = 0
                else:
                    background = 1
                    
                label = [date, death, background]
                tile_write(tile, count, name, save_path, label, tile_annot_path)
                count += 1
                tile_list.append(tile)
                bg_list.append(background)
    return tile_list, Thres_, w_num, h_num, bg_list

def tile_write(tile, count, name, save_path, label, annot_path):
    tile_name = name + '_' + str(count) + '.' + 'png'
    tile_path = os.path.join(save_path, tile_name)
    cv2.imwrite(tile_path, tile)
    
    csv_file = annot_path
    write_header = not os.path.exists(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(['Sample_Name', 'Date', 'Death', 'Background'])
        writer.writerow([name + '_' + str(count), label[0], label[1], label[2]])

def Image_Show(image):
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])
    plt.imshow(image / 255)
    # plt.axis('off')
    plt.show()
    
def Add_Border(image_list,Bold_Image_Size, bg_list):
    New_List_Image=[]
    for i, image in enumerate(image_list):
        if bg_list[i] == 0:
            img = cv2.copyMakeBorder(image, Bold_Image_Size, Bold_Image_Size, Bold_Image_Size, Bold_Image_Size,
                                            cv2.BORDER_CONSTANT,
                                            value=(255, 0, 0))
        else:
            img = cv2.copyMakeBorder(image, Bold_Image_Size, Bold_Image_Size, Bold_Image_Size, Bold_Image_Size,
                                            cv2.BORDER_CONSTANT,
                                            value=(0, 0, 255))
        New_List_Image.append(img)
    return New_List_Image

def Tile_Image(tile_list, image_shape, tile_size):
    # w_num = math.ceil(int(image_shape[0])/int(image_size))
    # h_num = math.ceil(int(image_shape[1])/int(image_size))
    # print(len(image_list))
    #
    h_num = int(image_shape[0])
    w_num = int(image_shape[1])
    image_shapes0 = int(h_num*tile_size)
    image_shapes1 = int(w_num*tile_size)
    image = np.zeros((image_shapes0, image_shapes1, 3), dtype=float, order='C')
    
    index = 0
    for i in range(h_num):
        tile_image = tile_list[index]
        for j in range(w_num):
            image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = tile_image
            index+=1
    return image


if __name__ == '__main__':
    # train_path = '../Image_train'
    # test_path = '../Image_test'
    train_path = '../../../../../mnt/d/peerasu/Image_train'
    test_path = '../../../../../mnt/d/peerasu/Image_test'
    annot_train_path = '../annotation/label_train.csv'
    annot_test_path = '../annotation/label_test.csv'
    
    # tile_train_path = '../Tile_train'
    # tile_test_path = '../Tile_test'
    tile_train_path = '../../../../../mnt/d/peerasu/Tile_train'
    tile_test_path = '../../../../../mnt/d/peerasu/Tile_test'
    tile_annot_train_path = '../annotation/tile_label_train.csv'
    tile_annot_test_path = '../annotation/tile_label_test.csv'
    
    tile_size = 224
    border_size = 10
    
    os.makedirs(tile_train_path, exist_ok=True)
    os.makedirs(tile_test_path, exist_ok=True)
    
    annot_train_file = pd.read_csv(annot_train_path)
    length_train = len(annot_train_file['Sample_Name'])
    
    annot_test_file = pd.read_csv(annot_test_path)
    length_test = len(annot_test_file['Sample_Name'])
    

    for i, name in enumerate(annot_train_file['Sample_Name']):
        image = cv2.imread(os.path.join(train_path, name + '.' + 'png'))
        tile_list, thres, w_num, h_num, bg_list = do_image_cut(image, tile_size, name, tile_train_path, annot_train_path, tile_annot_train_path)
        
        # # SHOW FULL TILE FROM PATCH
        # Image_Show(image)
        # Border_List = Add_Border(tile_list, border_size, bg_list)
        # # Tile_list = []
        # # for bord in Border_List:
        # #     if bg_list[i] == 0:
        # #         bord[bord == 0] = 255
        # #     Tile_list.append(bord)
        # new_tile_size = tile_size + border_size * 2
        # image_shape = [w_num, h_num]
        # Image_Tile = Tile_Image(Border_List, image_shape, new_tile_size)
        # Image_Show(Image_Tile)
        
    for i, name in enumerate(annot_test_file['Sample_Name']):
        image = cv2.imread(os.path.join(test_path, name + '.' + 'png'))
        tile_list, thres, w_num, h_num, bg_list = do_image_cut(image, tile_size, name, tile_test_path, annot_test_path, tile_annot_test_path)
        
        # # SHOW FULL TILE FROM PATCH
        # Image_Show(image)
        # Border_List = Add_Border(tile_list, border_size, bg_list)
        # new_tile_size = tile_size + border_size * 2
        # image_shape = [w_num, h_num]
        # Image_Tile = Tile_Image(Border_List, image_shape, new_tile_size)
        # Image_Show(Image_Tile)
        