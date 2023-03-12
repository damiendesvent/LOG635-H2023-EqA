from skimage.io import imshow, imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import closing
from skimage.measure import label, regionprops, regionprops_table
import pandas as pd
import numpy as np
import os
from IPython.display import display

#pip install scikit-image
#pip install -U scikit-learn scipy matplotlib
#pip install pandas
#pip install tqdm
#pip install ipython
#https://mattmaulion.medium.com/leaf-classification-an-image-processing-feature-extraction-approach-to-machine-learning-c0677e07da80

image_path_list = os.listdir("Labo_2/output/clean/Diamants/Diamant5")
df = pd.DataFrame()
for i in range(len(image_path_list)):
   
  image_path = image_path_list[i]
  image = rgb2gray(imread("Labo_2/output/clean/Diamants/Diamant5/"+image_path))
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
  table['label'] = image_path[5]
  df = pd.concat([df, table], axis=0)
df.head()
display(df)

X = df.drop(columns=['label', 'image', 'real_images'])
#features
X = X[['iqr','75th Percentile','inertia_tensor-1-1',
       'std_intensity','mean_intensity','25th Percentile',
       'minor_axis_length', 'solidity', 'eccentricity']]