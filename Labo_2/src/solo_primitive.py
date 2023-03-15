from skimage.io import imshow, imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import closing
from skimage.measure import label, regionprops, regionprops_table
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display

#pip install scikit-image
#pip install -U scikit-learn scipy matplotlib
#pip install pandas
#pip install tqdm
#pip install ipython
#pip install seaborn
#https://mattmaulion.medium.com/leaf-classification-an-image-processing-feature-extraction-approach-to-machine-learning-c0677e07da80

#image de test
#image = "Labo_2/output/clean/Cercles/Cercle5/29_Cercle5.jpg"

def solo_features(image):
    df = pd.DataFrame()
    image = rgb2gray(imread(image))

    binary = image < threshold_otsu(image)

    binary = closing(binary)
    label_img = label(binary)

    table = pd.DataFrame(regionprops_table(label_img, image,
                            ['convex_area', 'area', 'eccentricity',
                            'extent', 'inertia_tensor',                         
                            'major_axis_length', 'minor_axis_length',
                            'perimeter', 'solidity', 'image',
                            'orientation', 'moments_central',
                            'moments_hu', 'euler_number',
                            'equivalent_diameter',
                            'mean_intensity', 'bbox']))
    table['perimeter_area_ratio'] = table['perimeter']/table['area']
    real_images = []
    std = []
    mean = []
    percent25 = []
    percent75 = []

    for prop in regionprops(label_img): 
        min_row, min_col, max_row, max_col = prop.bbox
        img = image[min_row:max_row,min_col:max_col]
        real_images += [img]
        mean += [np.mean(img)]
        std += [np.std(img)]
        percent25 += [np.percentile(img, 25)] 
        percent75 += [np.percentile(img, 75)]

    table['real_images'] = real_images
    table['mean_intensity'] = mean
    table['std_intensity'] = std
    table['25th Percentile'] = mean
    table['75th Percentile'] = std
    table['iqr'] = table['75th Percentile'] - table['25th Percentile']
    table['label'] = 'img'
    df = pd.concat([df, table], axis=0)
    df.head()

    #features
    X = df[['area','extent', 'eccentricity','perimeter_area_ratio','std_intensity', 'mean_intensity','perimeter', 'solidity', 'convex_area', 'iqr', 'euler_number', 'equivalent_diameter', 'orientation','inertia_tensor-0-0',
            'moments_central-0-0','moments_central-0-2','moments_central-2-0','moments_central-2-2','moments_hu-2','moments_hu-6']]
    #affiche le tableau des features (une ligne est une forme trouvée)
    display(X)

    return X
