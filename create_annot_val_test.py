import pandas as pd

# Compute classes distribution
annot_train_file = pd.read_csv('../annotation_newnew/Annot_Val_Patient_WSI.csv')
dis = [0, 0]
for i, raw in annot_train_file.iterrows():
    if int(raw['Death']) == 0:
        dis[0] += 1
    else:
        dis[1] += 1


# Compute classes distribution
annot_train_file = pd.read_csv('../annotation_newnew/Annot_Test_Patient_WSI.csv')
for i, raw in annot_train_file.iterrows():
    if int(raw['Death']) == 0:
        dis[0] += 1
    else:
        dis[1] += 1
        
        
# Class frequencies
class_frequencies = [dis[0], dis[1]]
print(class_frequencies)
