import pandas as pd 


annot_val_path = '../annotation_newnew/Annot_Test_Tile.csv'
val_patient_path = '../annotation_newnew/Label_Test_WSI.csv'

annot_val_file = pd.read_csv(annot_val_path)
val_list = []
key_list = []
count_death = 0; count_survive = 0;
for i, raw in annot_val_file.iterrows():
    val_name = '_'.join(((raw['Sample_Name']).split('_'))[:-1])
    if val_name not in key_list:
        key_list.append(val_name)
        label = raw['Death']
        if label == 0:
            count_survive += 1
        else:
            count_death += 1
        val_list.append((val_name, label))
        
val_label = pd.DataFrame(val_list, columns=['Sample_Name', 'Death'])
val_label.to_csv(val_patient_path, index=False)
print(f'Survive: {count_survive}, Death: {count_death}')
        

