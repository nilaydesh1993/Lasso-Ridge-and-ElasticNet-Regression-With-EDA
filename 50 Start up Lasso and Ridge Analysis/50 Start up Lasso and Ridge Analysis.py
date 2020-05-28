"""
Created on Sat Mar 21 21:07:33 2020
By - Deshmukh
LASSO AND RIDGE
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pylab as plt
import pylab
import seaborn as sns

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split

# ===================================================================================================
# Business Problem - Performing the Lasso and Ridge Regression on 50_startups data to predict profit.
# ===================================================================================================

startups = pd.read_csv('50_Startups.csv')
startups.head()
startups.isnull().sum()
startups.info()
startups.columns = "RandD","admin","marketing","state","profit"
startups = startups.replace(" ","_",regex = True)

############################### - Exploratory Data Analysis - ###################################### 

# Summary
startups.describe()

# Histogram
startups.hist(grid = False)

# Boxplot
startups.boxplot(patch_artist = True, grid = False, notch = True)   

# Normal Quantile-Quantile plot
stats.probplot(startups.RandD,dist = 'norm',plot = pylab) # Normal distribution
stats.probplot(startups.admin,dist = 'norm',plot = pylab) # Normal distribution
stats.probplot(startups.marketing,dist = 'norm',plot = pylab) # Normal distribution
stats.probplot(startups.profit,dist = 'norm',plot = pylab) # Normal distribution

# Pairplot
sns.pairplot(startups, corner = True, diag_kind = "kde")

# Heat map and Correlation Coifficient 
sns.heatmap(startups.corr(), annot = True, cmap = 'Blues')

###################################### - Data Preprocessing - #######################################

# Nomalization of data (as data contain binary value)
#startups.iloc[:,0:3] = normalize(startups.iloc[:,0:3])

# Converting Dummy variable,Removing old columns and adding new dummy column in datafram.
dummy = pd.get_dummies(startups.state,drop_first = True)
startups = startups.drop(['state'],axis = 1)
startups = pd.concat([dummy,startups],axis = 1)

##################################### - Splitting data - ############################################

# Splitting in X and y
X = startups.iloc[:,0:5]
y = startups.iloc[:,5]

# Splitting in Train and Test 
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 100)

##################################### - Losso Regression - ##########################################

### Running a LASSO Regressor of set of alpha values and observing how the R-Squared, train_rmse and test_rmse are changing with change in alpha values
train_rmse = []
test_rmse = []
R_sqrd = []
alphas = np.arange(0,500,1)
for i in alphas:
    LRM = Lasso(alpha = i,normalize=True,max_iter=500)
    LRM.fit(X_train,y_train)
    R_sqrd.append(LRM.score(X_train,y_train))
    train_rmse.append(np.sqrt(np.mean((LRM.predict(X_train) - y_train)**2)))
    test_rmse.append(np.sqrt(np.mean((LRM.predict(X_test) - y_test)**2)))
    
# Plotting Alpha vs Train and Test RMSE.
plt.scatter(x=alphas,y=R_sqrd);plt.xlabel("alpha");plt.ylabel("R_Squared")
plt.scatter(x=alphas,y=train_rmse);plt.xlabel("alpha");plt.ylabel("RMSE")
plt.scatter(x=alphas,y=test_rmse);plt.xlabel("alpha");plt.ylabel("RMSE")
plt.legend(("alpha Vs R_Squared","alpha Vs train_rmse","alpha Vs test_rmse"))


##Another Way of finding alpha value by using GV but above is best than this
#from sklearn.model_selection import GridSearchCV
#lasso=Lasso()
#params={'alpha':np.arange(0,500,10)}
#Regressor=GridSearchCV(lasso,params,scoring='neg_mean_squared_error',cv=10)
#Regressor.fit(X,y)
##Print best parameter and score
#print('best parameter: ', Regressor.best_params_)
#print('best score: ', -Regressor.best_score_)


# Preparing Lasso Regression by considering alpha = 150 from above
LassoM1 = Lasso(alpha = 150, normalize=True)
LassoM1.fit(X_train,y_train)

# Parameters of model
LassoM1.coef_
LassoM1.intercept_

# Adjusted R-Squared value 
LassoM1.score(X_train,y_train) # 0.96

# Predication on Train and Test 
pred_train_lasso = LassoM1.predict(X_train)
pred_test_lasso = LassoM1.predict(X_test)

# Train and Test RMSE value
np.sqrt(np.mean((pred_train_lasso-y_train)**2)) # 7900
np.sqrt(np.mean((pred_test_lasso-y_test)**2)) # 12660

# Important Coefficient Plot
important_coff = pd.Series(LassoM1.coef_,index = X.columns)
important_coff.plot(kind = 'barh')

##################################### - Ridge Regression - ##########################################

### Running a Ridge Regressor of set of alpha values and observing how the R-Squared, train_rmse and test_rmse are changing with change in alpha values
train_rmse = []
test_rmse = []
R_sqrd = []
alphas = np.arange(0,2,0.02)
for i in alphas:
    RM = Ridge(alpha = i,normalize=True,max_iter=500)
    RM.fit(X_train,y_train)
    R_sqrd.append(RM.score(X_train,y_train))
    train_rmse.append(np.sqrt(np.mean((RM.predict(X_train) - y_train)**2)))
    test_rmse.append(np.sqrt(np.mean((RM.predict(X_test) - y_test)**2)))
    
# Plotting Alpha vs Train and Test RMSE.
plt.scatter(x=alphas,y=R_sqrd);plt.xlabel("alpha");plt.ylabel("R_Squared")
plt.scatter(x=alphas,y=train_rmse);plt.xlabel("alpha");plt.ylabel("RMSE")
plt.scatter(x=alphas,y=test_rmse);plt.xlabel("alpha");plt.ylabel("RMSE")
plt.legend(("alpha Vs R_Squared","alpha Vs train_rmse","alpha Vs test_rmse"))

# Preparing Ridge Regression by considering alpha = 0.01 from above
RidgeM1 = Ridge(alpha = 0.01, normalize=True)
RidgeM1.fit(X_train,y_train)

# Parameters of model
RidgeM1.coef_
RidgeM1.intercept_

# Adjusted R-Squared value 
RidgeM1.score(X_train,y_train) # 0.96

# Predication on Train and Test 
pred_train_ridge = RidgeM1.predict(X_train)
pred_test_ridge = RidgeM1.predict(X_test)

# Train and Test RMSE value
np.sqrt(np.mean((pred_train_ridge-y_train)**2)) # 7582
np.sqrt(np.mean((pred_test_ridge-y_test)**2)) # 13435

# Importanat Coefficient Plot
important_coff = pd.Series(RidgeM1.coef_,index = X.columns)
important_coff.plot(kind = 'barh',color = 'g')

##################################### - Elastic Net Regression - ##########################################

### Running a Elastic Net Regressor of set of alpha values and observing how the R-Squared, train_rmse and test_rmse are changing with change in alpha values
train_rmse = []
test_rmse = []
R_sqrd = []
alphas = np.arange(0,1,0.01)
for i in alphas:
    EN = ElasticNet(alpha = i,normalize=True,max_iter=500)
    EN.fit(X_train,y_train)
    R_sqrd.append(EN.score(X_train,y_train))
    train_rmse.append(np.sqrt(np.mean((EN.predict(X_train) - y_train)**2)))
    test_rmse.append(np.sqrt(np.mean((EN.predict(X_test) - y_test)**2)))
    
# Plotting Alpha vs Train and Test RMSE.
plt.scatter(x=alphas,y=R_sqrd);plt.xlabel("alpha");plt.ylabel("R_Squared")
plt.scatter(x=alphas,y=train_rmse);plt.xlabel("alpha");plt.ylabel("RMSE")
plt.scatter(x=alphas,y=test_rmse);plt.xlabel("alpha");plt.ylabel("RMSE")
plt.legend(("alpha Vs R_Squared","alpha Vs train_rmse","alpha Vs test_rmse"))

# Preparing Elastic Net Regression by considering alpha = 0.01 from above
Elastic = ElasticNet(alpha = 0.01, normalize=True)
Elastic.fit(X_train,y_train)

# Parameters of model
Elastic.coef_
Elastic.intercept_

# Adjusted R-Squared value 
Elastic.score(X_train,y_train) # 0.93

# Predication on Train and Test 
pred_train_elastic = Elastic.predict(X_train)
pred_test_elastic= Elastic.predict(X_test)

# Train and Test RMSE value
np.sqrt(np.mean((pred_train_elastic-y_train)**2)) # 10090
np.sqrt(np.mean((pred_test_elastic-y_test)**2)) # 14862

# Importanat Coefficient Plot
important_coff = pd.Series(Elastic.coef_,index = X.columns)
important_coff.plot(kind = 'barh',color = 'r')


# Form Above Three model Lasso is giving us best result so we can be use it for future prediction.


                    # ---------------------------------------------------- #


