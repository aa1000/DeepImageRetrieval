from shutil import copyfile
import os
import numpy as np
import DeepImageUtils as IU

def MakeDirChecked(path):
    if not os.path.isdir(path):
        os.mkdir(path)

root_folder = input('Enter the path of the image dataset root folder: \n')
root_folder = os.path.realpath(root_folder)

if not os.path.isdir(root_folder):
    print('no such folder:', root_folder)
    exit(1)

database_path = input('Enter the path to create the root folder of the database in: \n')
database_path = os.path.join(database_path, 'database')
database_path = os.path.realpath(database_path)

# get the path to all image files in the root folder
image_paths = IU.GetAllImagesInPath(root_folder)

# the root create database folder
MakeDirChecked(database_path)
for img_path in image_paths:
    # predict the categories of every image in the dataset 
    categories = IU.PredictImageCategory(img_path)

    # create a folder for every category to easily separate the data reducing search times
    for category in categories:
        category_path =  os.path.join(database_path, category) + '/'
        # create the directory for the category
        MakeDirChecked(category_path)
        # copy the image to the category folder
        # could potentially just save the extracted features there and have a reference to the place of the orignal image path
        # having a reference might help with avoiding duplicates and copy times if you don't need a backup
        # and if you are scapping the web you can just link to the original image in the features database to retrun that link later
        copy_path = os.path.join(category_path, IU.Path2Name(img_path))
        copyfile(img_path, copy_path)

# get all the created category folders in the database directory
category_folders = IU.GetAllFolderInPath(database_path)

# for every category create the feature vector for the images of that category present in the category file
for category_folder in category_folders:
    feature_vectors = []
    # get all image file paths' in the category's folder
    database_image_paths = IU.GetAllImagesInPath(category_folder)
    
    # extract the features of each image and append it to the feature list of that category
    # this is where you would put the link or the other path of the image instead of the local one
    for database_image_path in database_image_paths:
        img_features = [database_image_path, IU.CreateImageFeaturesVector(database_image_path)]
        feature_vectors.append(img_features)
  
  
    # save the features as a compressed numpy array object with the name of the category
    feature_file_name = IU.Path2Name(category_folder[:-1]) + '.npz'
    features_path = os.path.join(category_folder, feature_file_name)
    np.savez_compressed(features_path, feature_vectors)