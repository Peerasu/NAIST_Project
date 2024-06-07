import pandas as pd

# Compute classes distribution
annot_train_file = pd.read_csv('../annotation_newnew/Label_Train_SS.csv')
dis = [0, 0, 0]
id_list = []
for i, raw in annot_train_file.iterrows():
    if int(raw['Label']) == 0:
        dis[0] += 1
    elif int(raw['Label']) == 1:
        dis[1] += 1
    else:
        dis[2] += 1
        
    id = raw['Sample_Name'].split('_')[0]
    if id not in id_list:
        id_list.append(id)
    
# Class frequencies
class_frequencies = [dis[0], dis[1], dis[2]]
print(class_frequencies)
print(f'id: {len(id_list)}')



# # Compute classes distribution at patch-level for all dataset
# annot_train_file = pd.read_csv('../annotation_new/Label_Train_BL.csv')
# dis = [0, 0, 0]
# for i, raw in annot_train_file.iterrows():
#     if int(raw['Label']) == 0:
#         dis[0] += 1
#     elif int(raw['Label']) == 1:
#         dis[1] += 1
#     else:
#         dis[2] += 1

# annot_train_file = pd.read_csv('../annotation_new/Label_Test.csv')
# for i, raw in annot_train_file.iterrows():
#     if int(raw['Label']) == 0:
#         dis[0] += 1
#     elif int(raw['Label']) == 1:
#         dis[1] += 1
#     else:
#         dis[2] += 1
        
# annot_train_file = pd.read_csv('../annotation_new/Label_Val.csv')
# for i, raw in annot_train_file.iterrows():
#     if int(raw['Label']) == 0:
#         dis[0] += 1
#     elif int(raw['Label']) == 1:
#         dis[1] += 1
#     else:
#         dis[2] += 1
        
# class_frequencies = [dis[0], dis[1], dis[2]]
# print(class_frequencies)
