# --------------
# Code starts here
# import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns 
#### Data 1
# Load the data
auto_df = pd.read_csv(path)

# Overview of the data
print(auto_df.info())
print(auto_df.describe())

# Histogram showing distribution of car prices
sns.distplot(auto_df.price)

# Countplot of the make column
sns.countplot(auto_df.make)

# Jointplot showing relationship between 'horsepower' and 'price' of the car
sns.jointplot('horsepower','price',auto_df)

# Correlation heat map
sns.heatmap(auto_df.corr())

# boxplot that shows the variability of each 'body-style' with respect to the 'price'
sns.boxplot(auto_df['body-style'],auto_df['price'])

#### Data 2

# Load the data
spec_df = pd.read_csv(path2)

print(spec_df.describe())
print(spec_df.info())

# Impute missing values with mean
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='Nan',strategy='mean',axis=0)
spec_df.horsepower.replace('?','NaN',inplace=True)

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
imp.fit_transform(spec_df.horsepower[:,np.newaxis])

# Skewness of numeric features
df_num = spec_df.select_dtypes(exclude='object')
c = df_num.columns
from scipy.stats import skew,norm 

for col in c:
    if skew(spec_df[col]) > 1:
        spec_df[col] = np.sqrt(spec_df[col]) 
sns.distplot(spec_df['engine-size'])
# Label encode
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
df_cats = spec_df.select_dtypes(include='object',exclude=['float64'])
df_cats = df_cats.iloc[:,:-1]
df_cats = df_cats.iloc[:,1:]
df_cats.apply(LabelEncoder().fit_transform)
print(df_cats.head())
spec_df['area']=spec_df.height*spec_df.width
print(spec_df.head())


# Code ends here


