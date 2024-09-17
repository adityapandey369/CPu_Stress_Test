#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# INSTALL ALL THE LIBRARIES BEFORE STRESS TEST


# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install pandas==1.5.3\n!pip install tsfel\n')


# In[ ]:


get_ipython().run_cell_magic('capture', '', '\nimport os\nimport glob\nfrom scipy.stats import pearsonr\nimport pandas as pd\nimport numpy as np\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n%matplotlib inline\npd.show_versions()\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestRegressor\nfrom sklearn.metrics import mean_squared_error,mean_absolute_percentage_error, r2_score, mean_absolute_error\n')


# In[ ]:


df = pd.read_csv("/Users/adityapandey/Desktop/Coursera/dataset_Statistical.csv")


# In[ ]:


imp_features = ['2_ECDF Percentile_1',
 '2_Mean',
 '2_Interquartile range',
 '3_Skewness',
 '2_Standard deviation',
 '2_Median absolute deviation',
 '2_Mean absolute deviation',
 '2_Variance',
 '2_Max',
 '2_Root mean square',
 '3_Max',
 '2_Skewness',
 '3_Standard deviation',
 '3_Variance',
 '3_Root mean square',
 '3_Mean absolute deviation',
 '1_ECDF Percentile_0',
 '3_Interquartile range',
 '3_Median absolute deviation',
 '0_Histogram_5',
 '5_ECDF Percentile_0',
 '1_Mean',
 '7_Median absolute deviation',
 '1_Median',
 '1_Histogram_9',
 '1_Kurtosis',
 '3_ECDF Percentile_1',
 '5_Histogram_3']


# In[ ]:


selected_features = df[imp_features].copy()
selected_features


# In[ ]:


from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV


# In[ ]:


y = df.iloc[:,-1:]
x = selected_features
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 28 )


# In[ ]:


import time
start_time = time.time()

test_accuracy_rates = []

for d in range(9, 10):
  for a in range(21, 22):
    for c in range(40, 50):
      b = a/100
      x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=b, random_state=c)
      rf = RandomForestRegressor(n_estimators=500, max_depth=d, n_jobs=-1)
      rf.fit(x_train,y_train)

      y_pred = rf.predict(x_test)


      A2 = r2_score(y_test,y_pred)
      accuracy = (A2 * 100)
      print("Accuracy for" +str(d)+ "and"+ str(b)+ "and" +str(c)+ "is" ,accuracy)

      test_accuracy_rates.append(A2)

test_accuracy_rates

end_time = time.time()


# In[ ]:


elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.6f} seconds")


# In[ ]:


test_accuracy_rates.sort(reverse=True)
print(test_accuracy_rates)


# In[ ]:





# In[ ]:




