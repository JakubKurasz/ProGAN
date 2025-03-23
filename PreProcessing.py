import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import os
import cv2
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
from torchvision.transforms import InterpolationMode

def preProcessing(imgSize):
    LABELS_PATH = '/content/drive/My Drive/OCTDL_labels.csv'
    INPUT_PATH = '/content/drive/My Drive/OCTDL'
    OUTPUT_PATH = '/content/drive/My Drive/augmented_data'

    df = pd.read_csv(LABELS_PATH)

    dimensions = pd.DataFrame(df, columns = ['file_name', 'image_width', 'image_hight'])

    dimensions['z_width'] = zscore(dimensions['image_width'])
    dimensions['z_height'] = zscore(dimensions['image_hight'])

    z_threshold = 2.5

    filtered_dimensions = dimensions[
        (abs(dimensions['z_width']) <= z_threshold) &
        (abs(dimensions['z_height']) <= z_threshold)
    ].copy()


    filtered_dimensions['aspect_ratio'] = filtered_dimensions['image_width'] / filtered_dimensions['image_hight']
    avg_aspect_ratio = filtered_dimensions['aspect_ratio'].mean()

    height = filtered_dimensions['image_hight'].min()
    width = round(height * avg_aspect_ratio)
    target_size = (width, height)

    def resize_and_save(classname):
        input_dir = INPUT_PATH + '/' + classname
        output_dir = OUTPUT_PATH + '/' + classname

        os.makedirs(output_dir, exist_ok = True)

        for _, row in filtered_dimensions.iterrows():
                file_name = row['file_name'] + '.jpg'
                if classname.lower() not in file_name:
                    continue
                image_path = os.path.join(input_dir, file_name)

                # Read the image
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Error loading image {file_name}")
                    continue

                resized_img = cv2.resize(img, target_size, interpolation = cv2.INTER_AREA)

                output_path = os.path.join(output_dir, file_name)
                cv2.imwrite(output_path, resized_img)

    classnames = ['AMD', 'DME', 'ERM', 'NO', 'RAO', 'RVO', 'VID']

    #for classname in classnames:
        #resize_and_save(classname)


    # hyperparameters - to be adjusted
    TEST_TRAIN_SPLIT = 0.3
    BATCH_SIZE = 32
    print(imgSize[0])
    if imgSize[1] == 199:
      transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()

    ])
    else:
      transform = transforms.Compose([
          transforms.Resize((imgSize[1], imgSize[0]), interpolation=InterpolationMode.LANCZOS),
          transforms.Grayscale(num_output_channels=1),
          transforms.ToTensor()

      ])

    dataset = datasets.ImageFolder(OUTPUT_PATH, transform=transform)

    n_test = int(np.floor(TEST_TRAIN_SPLIT * len(dataset)))
    n_train = len(dataset) - n_test

    train_ds, test_ds = random_split(dataset, [n_train, n_test])

    train_dl = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
    test_dl = DataLoader(test_ds, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)


    # some useful info about the dataset
    print(f"Classes: {dataset.classes}")
    print(f"Number of training samples: {len(train_ds)}")
    print(f"Number of testing samples: {len(test_ds)}")
    #for i, (x, label) in enumerate(train_dl):
        #print(label)
        #break
        #print(f"Image shape: {x.shape}")
    return train_dl, test_dl, len(dataset)