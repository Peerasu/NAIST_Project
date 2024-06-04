import pandas as pd 

    
patch_annot_train_path = '../annotation_new/Patch_Annot_Train.csv'
patch_annot_test_path = '../annotation_new/Patch_Annot_Test.csv'
patch_annot_val_path = '../annotation_new/Patch_Annot_Val.csv'

true_train_label_path = '../annotation_new/Label_Train.csv'
true_test_label_path = '../annotation_new/Label_Test.csv'
true_val_label_path = '../annotation_new/Label_Val.csv'


annot_train_file = pd.read_csv(patch_annot_train_path)
count = [0, 0, 0]
for i, raw in annot_train_file.iterrows():
    if raw['Background'] == 0:
        if raw['Death'] == 0:
            count[0] += 1
        else:
            count[1] += 1
    else:
        count[2] += 1
print(f'Survive: {count[0]}, Death: {count[1]}, background: {count[2]}')


# annot_train_file = pd.read_csv(patch_annot_train_path)
# train_label_list = []
# for i, raw in annot_train_file.iterrows():
#     if raw['Background'] == 0:
#         label = int(raw['Death'])
#     else:
#         label = 2
#     train_label_list.append((raw['Sample_Name'], label))
# train_label = pd.DataFrame(train_label_list, columns=['Sample_Name', 'Label'])
# train_label.to_csv(true_train_label_path, index=False)


# annot_test_file = pd.read_csv(patch_annot_test_path)
# test_label_list = []
# for i, raw in annot_test_file.iterrows():
#     if raw['Background'] == 0:
#         label = int(raw['Death'])
#     else:
#         label = 2
#     test_label_list.append((raw['Sample_Name'], label))
# test_label = pd.DataFrame(test_label_list, columns=['Sample_Name', 'Label'])
# test_label.to_csv(true_test_label_path, index=False)


# annot_val_file = pd.read_csv(patch_annot_val_path)
# val_label_list = []
# for i, raw in annot_val_file.iterrows():
#     if raw['Background'] == 0:
#         label = int(raw['Death'])
#     else:
#         label = 2
#     val_label_list.append((raw['Sample_Name'], label))
# val_label = pd.DataFrame(val_label_list, columns=['Sample_Name', 'Label'])
# val_label.to_csv(true_val_label_path, index=False)
        

