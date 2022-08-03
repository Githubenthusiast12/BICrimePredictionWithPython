
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
import itertools
import squarify
import sklearn
from wordcloud import WordCloud, STOPWORDS
import scipy
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from scipy.spatial.distance import euclidean
import sys
from sklearn.preprocessing import MinMaxScaler
import math

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#Since dataset is too big, we initially selecting 10k rows for it
df= pd.read_csv('crime1.csv' ,encoding='latin-1', nrows = 100000)
df.drop("INCIDENT_NUMBER",axis=1, inplace=True) 
df[["DATE","TIME"]]=df['OCCURRED_ON_DATE'].str.split(" ",expand=True)
print(df.head())
print(df.info())
#removing any null values
df = df.dropna()

# Plot the histogram thanks to the distplot function
sns.distplot( a=df["YEAR"], hist=True, kde=False, rug=False )
  
# function to show the plot
plt.show()

def lineplt(x,y,xlabel,ylabel,title,size,tick_spacing):
    fig,ax=plt.subplots(figsize = size)
    plt.plot(x,y)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xlabel(xlabel,fontsize = 15)
    plt.ylabel(ylabel,fontsize = 15)
    plt.title(title,fontsize = 20)
plt.show()

# Create 2 columes DateFrame
def createdf(c1,d1,c2,d2):
    dic = {c1:d1,c2:d2}
    df = pd.DataFrame(dic)
    return df

# Put Date and Count into a new Dataframe
c = createdf("Date",df["DATE"].value_counts().index,"Count",df["DATE"].value_counts())


print('skewness is ' + str(c['Count'].skew()))
print('kurtosis is ' + str(c['Count'].kurt()))

bin=pd.cut(c["Count"],50)
fre= createdf("Bin",bin.value_counts().index,"Count",bin.value_counts())
fre_sort = fre.sort_values(by = "Bin", ascending = True)

(_,p) = scipy.stats.shapiro(fre_sort["Count"])
print('p-value is ' + str(p))

(_,p) = scipy.stats.kstest(fre_sort["Count"],'norm')
print('p-value is ' + str(p))

c=c.sort_values(by="Date",ascending = True)
lineplt(c["Date"],c["Count"],"Date","Count","Crimes by Time",(20,15),80)

plt.show()

comment_words = ''
stopwords = set(STOPWORDS)
 
# iterate through the csv file
for val in df.OFFENSE_DESCRIPTION:
     
    # typecaste each val to string
    val = str(val)
 
    # split the value
    tokens = val.split()
     
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
     
    comment_words += " ".join(tokens)+" "
 
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
 
# plot the WordCloud image                      
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()




model=SARIMAX(c['Count'], order=(1,1,1), seasonal_order=(1,1,1, 7)).fit()
summary = model.summary()
print(summary)
model.plot_diagnostics(figsize=(15, 12))
plt.show()

