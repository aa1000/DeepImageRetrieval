# Deep Image Retrieval
**Content-based image retrieval example using Deep Neural Networks (Inception Res-Net V2 with imgnet weights) in Keras, feel free to test the IPython notebook attached to test the code in google colab.**

This code will sort a dataset of search images into categories predicted by Inception Res-Net then create a feature vector for each image in the search database.
The feature vector is created using the conv layers of Inception Res-Net where the output tensor is then global average pooled and normalized resulting in a feature vector of size 1536.

When a query image is entered, it's converted into a feature vector and compared to all the feature vectors of images in its category using cosine similarity and outputting a sorted array of the most matching images.

Currently, it will give an error and exit if no matching image was found and you have to re-create the database after adding any new images. However, it should be possible to just append the features of the new images to existing ones and possibly add every query image to the database as well.

The database doesn't have to be local image files, the code can be tweaked to scrap images from the web and have the link of the image referenced in the saved feature vector instead of the local path to the image.


### **Dependencies:**

* **Keras**
* **numpy**
* **scipy**
* **matplotlib**
* **shutil**

### **Steps:**

* Create a directory and put the `'CreateDatabase.py'`, `'RetrieveSimilarImages.py'` and `'ImageUtils.py'` scripts in it

* Create a dataset of any number of images you want to use as a database and move them into a new root directory(s)

* Run `'CreateDatabase.py'` which will prompt you to enter the dataset path

* Choose the location where you would like the root directory of the database to be

* CreateDatabase search for all images in the dataset directory recursively and:
    * Create a sub directory in the root database directory for every category of images present in the dataset (predicted by inception resnet)
    * Make a copy of every image in it's predicted category category folder (each image has 4 categories and thus will have 4 copies in different folders)
    * For every category, create the feature vectors for all images and save them in a compressed npz file in each directory

* Run `'RetrieveSimilarImages.py'` which will prompt you to choose an image file to use as the query image then show the closest matches on a pyplot


### **Observations:**
This is much faster compared to [Classical Content-based Image Retrieval](https://github.com/aa1000/ImageRetrievalClassic) as it only searches the images in 4 directories instead of all images in the database. Those images have a very high chance of being of similar to the query image as well.

The feature vector is also much smaller than the classical approach (while providing better results!) so it can be saved in much less space and the distance calculations are much faster.

Content-based Image Retrieval using deep neural networks is overall much more accurate and faster both in creating the database and search times but it consumes a huge amount of GPU power and memory.

### **Possible Future Improvements:**
* Reducing the NN model size
* Reducing the size of the output features vector maybe using some sort of a dimensional reduction algorithm
* Providing a faster search using a better vector search algorithm like the one Microsoft released
