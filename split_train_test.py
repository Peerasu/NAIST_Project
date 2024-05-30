import os
import random
import pandas as pd 
import csv
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

        
def create_40_mag_image(name, mag, data_path, save_path):
    image = cv2.imread(os.path.join(data_path, name+'.tif'))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # ถ้าใช้ cv2 ต่อ (ปกติก็ใช้ต่อ) ไม่ต้องปรับ cvtColor, ปรับตอน show ด้วย plt เท่านั้น (แล้วเปลี่ยนกลับเป็น BGR ด้วย ถ้าใช้ cv2 ต่อ)
    # ไม่ต้องปรับ range จาก 0-255 (opencv) เป็น 0-1 (plt) เลย เพราะ เวลา show ด้วย plt, library plt จะรู้เองว่าตอนนี้เป็น 0-255 เเละจะโชว์ในรูป 0-1 โดยอัตโนมัติ (แต่ตัว image ยังเป็น range 0-255 อยู่เหมือนเดิม)
    
    image_shape = image.shape
    factor = int(math.ceil(40 / mag))
    h_length = int(math.ceil(image_shape[0] / factor))
    w_length = int(math.ceil(image_shape[1] / factor))
    image = cv2.resize(image, dsize=(w_length*factor, h_length*factor), interpolation=cv2.INTER_LINEAR)
    
    tile_list = []
    
    count = 0
    for i in range(factor):
        for j in range(factor):
            new_image = image[i*h_length:(i+1)*h_length, j*w_length:(j+1)*w_length]
            new_image_name = name + f'_{count}.png'
            new_image_path = os.path.join(save_path, new_image_name)

            new_image = cv2.resize(new_image, dsize=(3360, 3360), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(new_image_path, new_image)
            tile_list.append(new_image)
            count += 1
    return count, tile_list, factor
            

def create_label_for_40_mag(name, count, date, death, annot_40_path):
    csv_file = annot_40_path
    for i in range(count):
        write_header = not os.path.exists(annot_40_path)
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(['Sample_Name', 'Date', 'Death'])
            writer.writerow([name + f'_{i}', date, death])
            
            
            
def Image_Show(image):
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])
    plt.imshow(image / 255)
    # plt.axis('off')
    plt.show()
    
def Add_Border(image_list,Bold_Image_Size):
    New_List_Image=[]
    for i, image in enumerate(image_list):
        img = cv2.copyMakeBorder(image, Bold_Image_Size, Bold_Image_Size, Bold_Image_Size, Bold_Image_Size,
                                        cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))
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
    data_path = '../Image'
    annot_path = '../annotation/label.csv'
    # train_data_path = '../Image_train'
    # test_data_path = '../Image_test'
    train_data_path = '../../../../../mnt/d/peerasu/Image_train'
    test_data_path = '../../../../../mnt/d/peerasu/Image_test'
    train_annot_path = '../annotation/label_train.csv'
    test_annot_path = '../annotation/label_test.csv'
    
    os.makedirs(train_data_path, exist_ok=True)
    os.makedirs(test_data_path, exist_ok=True)
    
    annot_file = pd.read_csv(annot_path)
    length = len(annot_file['filename'])
    index = np.arange(length)
    random.shuffle(index)
    
    split_ratio = 0.8
    train_index = index[:math.ceil(split_ratio*length)]
    test_index = index[math.ceil(split_ratio*length):]
    
    annot_train = annot_file.iloc[train_index]
    annot_test = annot_file.iloc[test_index]

    for i, name in enumerate(sorted(annot_train['filename'])):
        mag = int((name.split('_')[2]).split('HE')[-1])
        count, tile_list, factor = create_40_mag_image(name, mag, data_path, train_data_path)
        create_label_for_40_mag(name, count, annot_train['lastfu_date'].iloc[i], annot_train['lastfu_death'].iloc[i], train_annot_path)
        
        # SHOW FULL IMAGE FROM TILE
        tile_size = 224
        border_size = 2
        w_num = factor; h_num = factor
        
        image = cv2.imread(os.path.join(data_path, name + '.' + 'png'))
        Image_Show(image)
        
        Border_List = Add_Border(tile_list, border_size)
        new_tile_size = tile_size + border_size * 2
        image_shape = [w_num, h_num]
        Image_Tile = Tile_Image(Border_List, image_shape, new_tile_size)
        Image_Show(Image_Tile)
    
    for i, name in enumerate(sorted(annot_test['filename'])):
        mag = int((name.split('_')[2]).split('HE')[-1])
        count, image_list, factor = create_40_mag_image(name, mag, data_path, test_data_path)
        create_label_for_40_mag(name, count, annot_test['lastfu_date'].iloc[i], annot_test['lastfu_death'].iloc[i], test_annot_path)
        
        # SHOW FULL IMAGE FROM TILE
        tile_size = 224
        border_size = 2
        w_num = factor; h_num = factor
        
        image = cv2.imread(os.path.join(data_path, name + '.' + 'png'))
        Image_Show(image)
        
        Border_List = Add_Border(tile_list, border_size)
        new_tile_size = tile_size + border_size * 2
        image_shape = [w_num, h_num]
        Image_Tile = Tile_Image(Border_List, image_shape, new_tile_size)
        Image_Show(Image_Tile)
    
