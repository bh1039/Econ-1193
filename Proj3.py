# Importing necessary libraries for data handling and mathematical operations
import pandas as pd
import numpy as np
import math

# Importing plotting libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Importing statsmodels for regression analysis
import statsmodels.formula.api as smf

# Reading the dataset
url = "https://github.com/ArieBeresteanu/Econ1193_Spring2024/raw/main/cardata2005.csv"
cars = pd.read_csv(url)

# Displaying the first few rows of the DataFrame to inspect the data
cars.head()

# Getting basic information from data
nrows, ncols = cars.shape
print(f"There are {nrows} rows/cars and {ncols} columns.", "\n \n Data Frame Info:")
# Displaying Dataframe column information
print(cars.info())

# Using a lambda function to map 0,2,3, and 4 as numerical categories for cars
cars['category'] = cars['segm1'].map(lambda x: math.floor((x)/10))

# Dictionary to map car names to numbers
categoryDict = {
    '0': 'passenger cars',
    '2': 'minivans',
    '3': 'SUV',
    '4': 'light trucks'   
}

# Creating a column category name to descibe the type of car 
cars['categoryName'] = cars['category'].map(lambda x: categoryDict[str(x)])

# Creating new columns to combine length and width, and to combine city and highway mpg
cars['mpg_combined'] = cars['mpg_city']*0.55 + cars['mpg_highway']*0.45
cars['footprint'] = cars['width'] * cars['length'] / 1000  # Rescaling for footprint calculation

# Generating descriptive statistics for all columns
stats_table = cars.describe()
# Dropping irrelevant columns for clarity
stats_table.drop(columns=['year', 'firm_id', 'segm1'], inplace=True)
# Transposing the table for clarity
stats_table.transpose()

# Counting how many cars are hybrid using 'value_counts' and creating a DataFrame from it
hybrid_counts = pd.DataFrame(cars['hybrid'].value_counts())
# Dictionary to label hybrids
categoryDict2 = {
   '0': 'Not Hybrid',
   '1': 'Hybrid',
}
# Applying the dictionary to give meaningful labels to hybrid counts
hybrid_counts['label'] = hybrid_counts.index.map(lambda x: categoryDict2[str(x)])
hybrid_counts

# Sorting cars by 'Quantity' and displaying top and bottom entries
top_cars = cars[['Quantity', 'model']].sort_values(by='Quantity', ascending=False)
top_cars.head(3)  # Top three selling cars
top_cars.tail(3)  # Bottom three selling cars

nHH2005 = 113343000  # Number of households in 2005 from external data from the excel file on professor Beresteanu's Github
totalQuanity = sum(cars["Quantity"])  # Summing to find total cars sold
S0 = 1 - totalQuanity/nHH2005  # Calculating the proportion of households that did not buy a car
logS0 = math.log(S0)  # Taking the logarithm of the proportion

# Calculating market share for each car, its logarithm, and defining a new variable 'Y' for regression
cars['marketShare'] = cars['Quantity']/nHH2005
cars['log_marketShare'] = cars['marketShare'].apply(math.log)
cars['Y'] = cars['log_marketShare'] - logS0

# Creating crosstabs and calculating distances to category averages for certain car characteristics
carCat = pd.crosstab(index=cars['categoryName'], columns='count')
characteristics = ['mpg_combined','footprint', 'hp', 'disp', 'weight']
cars['categoryCount'] = cars['categoryName'].map(lambda x: carCat.loc[x,'count'])
featuresAvg = cars.groupby(['categoryName'])[characteristics].mean()

#characteristics is a list of strings. Each string in the list is a name of a characteristicdef dist2Cat(characteristics):
    for ch in characteristics:
       # 1. expand                                          
        cars[ch+'Avg'] = cars['categoryName'].map(lambda x: featuresAvg[ch][x])
        cars[ch+'Avg'] = cars['categoryName'].map(lambda x: featuresAvg[ch][x]) # can use .loc here
        cars[ch+'Dist'] = cars[ch] - cars[ch+'Avg']

        # could do the next part in 1 line and not using lambda fn

        # 2. difference
        cars[ch+'Dist'] = cars[ch]-cars[ch+'Avg']
        # 3. square
         cars[ch+'Dist'] = cars[ch+'Dist'].map(lambda x: x*x)

dist2Cat(characteristics)

# Modified function to calculate adjusted differences accounting for own values in averages
def dist2CatV2(characteristics):
    for ch in characteristics:
        cars[ch+'Avg2'] = (cars[ch+'Avg']*cars['categoryCount'] - cars[ch])/(cars['categoryCount']-1)
        cars[ch+'Dist2'] = cars[ch] - cars[ch+'Avg2']
        cars[ch+'Dist2'] = cars[ch+'Dist2'].map(lambda x: x*x)

dist2CatV2(characteristics)

# Visualizing correlation matrix for selected features
features = ['Price', 'disp','hp', 'wheel_base', 'weight', 'mpg_combined', 'footprint', 'hpDist', 'dispDist', 'mpg_combinedDist', 'footprintDist']
fig = plt.figure(figsize = (9,9))
sns.heatmap(cars[features].corr(), annot=True, annot_kws={'size':10}, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()

# Linear regression models using statsmodels, fitting models, and displaying summaries
model = smf.ols(formula = "Price ~ hp + mpg_combined + footprint + weight + C(category) + mpg_combinedDist + footprintDist", data=cars).fit(cov_type='HC1')
print(model.summary())


model = smf.ols(formula = "Price ~ hp + mpg_combined + weight + mpg_combinedDist2 + C(category) + weightDist2", data=cars).fit(cov_type='HC1') # to make regression robust
print(model.summary())

# Predicting prices using the fitted model and performing a second stage regression
cars['Price_hat'] = model.predict()
secondStageV1 = smf.ols(formula='Y ~ hp + mpg_combined + weight + Price_hat + C(category)', data=cars).fit(cov_type='HC1')
print(secondStageV1.summary())

# Visual comparison of actual and predicted values for regression targets using distribution plots
import warnings
warnings.filterwarnings("ignore")
y_pred = secondStageV1.predict()
plt.figure(figsize = (7,5))
sns.distplot(cars['Y'], color = 'r', hist=False, label='Actual Values', kde_kws={"shade": False})
sns.distplot(y_pred, color = 'purple', hist=False, label='Predicted Values', kde_kws={"shade": False})
plt.title('Regression Distribution of Actual vs Predicted Values')
plt.legend()
plt.show()

# Calculating correlations of numeric columns with 'Y' and sorting by absolute values
corrs = cars.select_dtypes(include=np.number).astype('float').corr()
corrs['Y'].abs().sort_values(ascending=False)
