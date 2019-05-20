import numpy as np
import os
from platform import platform
from glob import iglob

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras.layers import  GlobalAveragePooling2D, Lambda
from keras.backend import l2_normalize
from keras.models import Model

# Full InceptionResNetV2 to use for classifiying and categorizing images to create the database
# We use the categories of inception net as a sort of a hierarchy to reduce search times and better retrieve images
inception_category = InceptionResNetV2(include_top=True, weights='imagenet')

# We also use the conv layers of InceptionResNetV2 without the fully connected outpuy layers to act as a feature extractor of our images
inception_conv = InceptionResNetV2(include_top=False, weights='imagenet')
# We add a GlobalAveragePooling layer after the conv layers to be able to reduce the the dimentionality of the output
# conv layers have an output of shape (1, 8, 8, 1536) and by taking the global average pooling of each channel so a (1, 1536) tensor
global_pooling = GlobalAveragePooling2D()(inception_conv.output)
# then we normalize the average pooled tensor so the results are more consistant when calculating the distance or KNN later
norm_lambda = Lambda(lambda  x: l2_normalize(x,axis=1))(global_pooling)
# the input is the normal InceptionResNetV2 input of (n, 299, 299, 3) and the output is the normalised features of the norm layer
feature_extractor = Model(inputs=[inception_conv.input], outputs=[norm_lambda])
# compile model (just to be able to use it? we don't require any training on the layers we added)
feature_extractor.compile(optimizer='rmsprop', loss='mse')


def LoadAndProcessImage(img_path):
    # load the image in the right size for the InceptionResNetV2 model
    img = image.load_img(img_path, target_size=(299, 299))
    # turn the image object into an RGB pixel array
    img = image.img_to_array(img)
    # Keras' Inception can work on a batch of images at a t a time so the input has to be 4D (n, width, hight, channels)
    img = np.expand_dims(img, axis=0)
    # preprocessing the image to be a valid input for Inception (0, 255) pixel values -> (-1, 1)
    return preprocess_input(img)

def PredictImageCategory(img_path):
    # load the image and prepare it to be a proper input for InceptionResNet
    img = LoadAndProcessImage(img_path)
    # predict the class of the image
    preds = inception_category.predict(img)
    # turn the output into understanable named categories
    # the model can predict the class of multiple images at once but here we would use only one image
    # so we can take only the first entry in the output list
    decoded_preds = decode_predictions(preds)[0]
    # return the string names of the categories only
    return [decoded_preds[i][1] for i in range(0, len(decoded_preds))]
    
# return a generator that iterates over all (supported) image file paths in the given path (path must end in /)
def GetAllImagesInPath(path):
    jpg_path = os.path.join(path, '**/*.jpg')
    jpeg_path = os.path.join(path, '**/*.jpeg')
    bmp_path = os.path.join(path, '**/*.bmp')
    png_path = os.path.join(path, '**/*.png')

    image_paths = []

    image_paths.extend( iglob(jpg_path, recursive=True) )
    image_paths.extend( iglob(jpeg_path, recursive=True) )
    image_paths.extend( iglob(bmp_path, recursive=True) )
    image_paths.extend( iglob(png_path, recursive=True) )
    
    # windows is case insensitive so we don't need to add this
    if not platform().startswith('Windows'):
        jpg_path = os.path.join(path, '**/*.JPG')
        jpeg_path = os.path.join(path, '**/*.JPEG')
        bmp_path = os.path.join(path, '**/*.BMP')
        png_path = os.path.join(path, '**/*.PNG')

        image_paths.extend( iglob(jpg_path, recursive=True) )
        image_paths.extend( iglob(jpeg_path, recursive=True) )
        image_paths.extend( iglob(bmp_path, recursive=True) )
        image_paths.extend( iglob(png_path, recursive=True) )

    return image_paths

def GetAllFolderInPath(path):
    query = os.path.join(path, '**/')
    return iglob(query, recursive=True)

def GetAllFeaturesInPath(path):
    query = os.path.join(path, '**/*.npz')
    return iglob(query, recursive=True)

def OpenImage(img_path):
    # load the image object
    img = image.load_img(img_path)
    # turn the image object into an RGB pixel array and then into a float array so PyPlot can read it
    return image.img_to_array(img)/255.

def Path2Name(img_path):
    return img_path.split('/')[-1]

def CreateImageFeaturesVector(img_path):
    
    img = LoadAndProcessImage(img_path)
    features_vector = feature_extractor.predict(img)

    return features_vector.flatten()



