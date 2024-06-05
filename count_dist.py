import pandas as pd

# Compute classes distribution
annot_train_file = pd.read_csv('../annotation_new/Label_Train_BL.csv')
dis = [0, 0, 0]
for i, raw in annot_train_file.iterrows():
    if int(raw['Label']) == 0:
        dis[0] += 1
    elif int(raw['Label']) == 1:
        dis[1] += 1
    else:
        dis[2] += 1
        
# Class frequencies
class_frequencies = [dis[0], dis[1], dis[2]]
print(class_frequencies)