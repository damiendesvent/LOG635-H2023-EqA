from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import closing
from skimage.measure import label, regionprops, regionprops_table
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
from IPython.display import display
import seaborn as sn

#pip install scikit-image
#pip install -U scikit-learn scipy matplotlib
#pip install pandas
#pip install tqdm
#pip install ipython
#pip install seaborn
#https://mattmaulion.medium.com/leaf-classification-an-image-processing-feature-extraction-approach-to-machine-learning-c0677e07da80
#1158,808, 1966
image_path_list = os.listdir("Labo_2/output/clean/Cercles")
dataframe = pd.DataFrame()
colors = {'Cercle2':'red', 'Cercle5':'blue'}
for k in (image_path_list):
  image_list = os.listdir("Labo_2/output/clean/Cercles/"+str(k))
  print(k)
  for i in  range(len(image_list)):
    image_path = image_list[i]
    image = rgb2gray(imread("Labo_2/output/clean/Cercles/"+k+"/"+image_path))
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
    table['label'] = k
    dataframe = pd.concat([dataframe, table], axis=0)
  
dataframe.head()
#display(df)
dataframe.plot.scatter(x='equivalent_diameter',y='perimeter',c= dataframe['label'].map(colors))
#plt.show()
temp = dataframe[['area','extent', 'eccentricity','perimeter_area_ratio','std_intensity', 'mean_intensity','perimeter', 'solidity', 'convex_area', 'iqr', 'euler_number', 'equivalent_diameter', 'orientation','inertia_tensor-0-0',
           'moments_central-0-0','moments_central-0-2','moments_central-2-0','moments_central-2-2','moments_hu-2','moments_hu-6']]
temp2 = dataframe[['bbox-0','bbox-1','bbox-2','bbox-3']]
corr_matrix = temp.corr()
#sn.heatmap(corr_matrix, annot=True)
#print(df.columns.tolist())
plt.show()

X = dataframe.drop(columns=['label', 'image', 'real_images'])
#features
X = X[['iqr','75th Percentile','inertia_tensor-1-1',
       'std_intensity','mean_intensity','25th Percentile',
       'minor_axis_length', 'solidity', 'eccentricity']]
display(X)