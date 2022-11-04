from tqdm import tqdm
import csv
import os
import numpy as np
from PIL import Image

def save_csv(target_list, output_file_name):
    """
    将数据写入csv文件
    """
    if not output_file_name.endswith('.csv'):
        output_file_name += '.csv'
    csv_file = open(output_file_name, "w", newline="")
    key_data = target_list[0]
    value_data = [target for target in target_list]
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(key_data)
    csv_writer.writerows(value_data)
    csv_file.close()

all_data = []
# open annotation file
input_folder="fashion-product-images"
output_folder="train-val"

with open("fashion-product-images/styles.csv",'r') as csv_file:
    # parse it as CSV
    reader = csv.DictReader(csv_file)
    # tqdm shows pretty progress bar
    # each row in the CSV file corresponds to the image
    for row in tqdm(reader, total=reader.line_num):
        # we need image ID to build the path to the image file
        img_id = row['id']
        # we're going to use only 3 attributes
        gender = row['gender']
        articleType = row['articleType']
        baseColour = row['baseColour']
        img_name = os.path.join(input_folder, 'images', str(img_id) + '.jpg')
        # check if file is in place
        if os.path.exists(img_name):
            # check if the image has 80*60 pixels with 3 channels
            img = Image.open(img_name)
            if img.size == (60, 80) and img.mode == "RGB":
                all_data.append([img_name, gender, articleType, baseColour])
        else:
            print("Something went wrong: there is no file ", img_name)

# set the seed of the random numbers generator, so we can reproduce the results later
np.random.seed(42)
# construct a Numpy array from the list
all_data = np.asarray(all_data)

# Take 40000 samples in random order
inds = np.random.choice(40000, 40000, replace=False)
# split the data into train/val and save them as csv files

save_csv(all_data[inds][:32000], os.path.join(output_folder, 'train.csv'))
save_csv(all_data[inds][32000:40000], os.path.join(output_folder, 'val.csv'))
