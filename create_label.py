import pandas as pd 

    
tile_annot_train_path = '../annotation/tile_label_train_4.csv'
tile_annot_test_path = '../annotation/tile_label_test_4.csv'
true_train_label_path = '../annotation/true_train_label_4.csv'
true_test_label_path = '../annotation/true_test_label_4.csv'

annot_train_file = pd.read_csv(tile_annot_train_path)
train_label_list = []
for i, raw in annot_train_file.iterrows():
    if raw['Background'] == 0:
        label = int(raw['Death'])
    else:
        label = 2
    train_label_list.append((raw['Sample_Name'], label))
train_label = pd.DataFrame(train_label_list, columns=['Sample_Name', 'Label'])
train_label.to_csv(true_train_label_path, index=False)


annot_test_file = pd.read_csv(tile_annot_test_path)
test_label_list = []
for i, raw in annot_test_file.iterrows():
    if raw['Background'] == 0:
        label = int(raw['Death'])
    else:
        label = 2
    test_label_list.append((raw['Sample_Name'], label))
test_label = pd.DataFrame(test_label_list, columns=['Sample_Name', 'Label'])
test_label.to_csv(true_test_label_path, index=False)
        

