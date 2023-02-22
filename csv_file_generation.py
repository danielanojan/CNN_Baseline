import os
import csv

root_dir = '/home/daniel/simase_network/Cleft_lip_data/test'
label_list = ['healthy', 'mild', 'severe']



with open('/home/daniel/simase_network/Cleft_lip_data/test/test_file.csv', mode='w') as file:
    file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    file_writer.writerow(["Filen_name", "label", "class"])
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:

            filename = (os.path.join(subdir, file))
            try:
                class_val = subdir.split('/')[-1]

                index =label_list.index(class_val)
                file_writer.writerow([file, index ,class_val])
            except:
                print ("Exception Occured")
                continue



