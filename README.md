Seattle Airbnb Homestays
Price Prediction
Karthik Garimella, Sandeep Alfred
MA5790 Predictive Modelling
Abstract
Seattle is a city surrounded by water, mountains and evergreen forests which is also home to the
Space Needle[5]
. Airbnb homestays are available all around the world with each bringing their
unique interpretation of a cozy, comfortable and congenial place. Homestays in Seattle would be
an enticing option for people visiting and the prices of homestays matters for the experience it
might provide. Airbnb is a public company open about their data which lends a helping hand in
predicting the price of a homestay. The main goal of the study is to predict the prices of Airbnb
homestays in Seattle to understand how volatile the prices could be and determine if the models
trained on the open data set. This study helps in providing the distribution of prices in a high-cost
of living area and the ability to settle on a reasonable price for new homestays in the region. This
study will focus on how the original data looks like, how it was transformed for model building
with both linear and non-linear models being considered. The dataset would be split in train and
test with cross validation resampling. The best models trained on the training sample will be run
on the test set which will assist in selecting the best model and also the most important predictors
useful in deducing the price of an Airbnb homestay in Seattle, USA.


1. Background
Airbnb is an American company which operates in the short and long term homestays of
experiences all over the world. Each homestay brings their own unique style with some homes
having a pizzazz look, some traditional, some historic. Airbnb is an abbreviation of its original
name: “Airbed and Breakfast”[1]
. It acts as a broker and charges a commission from each
booking. Airbnb was founded in 2008 by Brian Chesky, Nathan Blecharczyk, and Joe Gebbia. It
is the best-known company for short-term housing[2][3]
. Washington state has restrictions on how
hosts must obtain licenses and cannot rent more than two units[4]
. This could impact on how the
prices could are set for the homestays. Predicting prices could help other potential Airbnb hosts
to land on a reasonable price for their homestays in Seattle. Figure 1 shows the geospatial
locations of the airbnb homestays used in this project.

![image](https://github.com/user-attachments/assets/00cda9bf-a772-4fea-acd6-662a077402b6)

2. Variable Introduction and Definitions:
The data has been retrieved from the Airbnb open dataset available for use by the public[6]
. The
Seattle dataset consists of 6442 samples and 77 predictors. There are a few samples where the
response variable does not exist, i.e., the price of the Airbnb homestay is unavailable. After
removing the Null response samples from the dataset, there are 6011 samples available for the
data preprocessing. The dataset also consists of predictors which are either irrelevant or not
feasible to convert the predictor as a dummy variable or a numerical value. These predictors
consist of information about when the data was scraped, description of the Airbnb homestay, url
for the homestay and host profile picture. After removing such predictors, we are left with 28
predictors with 14 numerical, 6 categorical, 5 logical and 3 date predictors in our dataset. T

![image](https://github.com/user-attachments/assets/50443ce2-05cd-4f55-8ee4-d143204377af)

4. Preprocessing of the predictors
The dataset we have currently consists of 6011 samples and 28 predictors. The logical columns
are converted to True/False, i.e., 0/1. The numerical columns are kept as and the date predictors
are converted to a date data type with day, month and year being split into separate columns. The
month predictors only have 12 unique values which are converted into a factor data type
considering it as a categorical variable. The same applies to the predictor bedrooms which
comprises only 10 unique values. Regarding all the categorical variables, they are converted into
dummy variables which increases our predictor size to 75. The degenerate columns are removed
from the set of predictors which leaves us with 54 predictors.
Figure 2. shows there are missing values in some of the predictor samples which models cannot
handle. A KNN imputation with k = 5 is performed on the dataset which will use the nearest
neighbors to impute the missing values.

![image](https://github.com/user-attachments/assets/58b53b0f-42dd-491e-b15b-198f09642e81)



b. Transformations
The data is first preprocessed by centering and scaling. Before splitting the data for model fitting,
the distribution of the predictors is analyzed. Most of the predictors are highly skewed (i.e., a
skewed distribution of the predictor data, shown in figure 4) which warrants a transformation. A
YeoJohnson transformation is performed to normalize the data. The YeoJohnson transformation
can handle predictors with negative values, hence it was preferred in the place of BoxCox. Since
our response variable was also highly skewed, we transformed the response variable using
YeoJohnson transformation. Figure 5 shows how much the data distribution has improved after
the transformation was applied.

![image](https://github.com/user-attachments/assets/714da45d-f870-4276-a50d-6dda563c8ad0)


![image](https://github.com/user-attachments/assets/8a82607f-e4fc-4ea2-bfe4-96b29998d644)

The existence of outliers also degrades the performance of the few models that the data is being
trained on which requires a spatial sign transformation. The spatial sign transformation helps in
minimizing the effect of outliers which might skew the model.

After performing the above transformations, the skewness value of each predictor is analyzed to
examine how the transformations affect the predictor distribution, figures 7 and 8 show the
improvement in outliers after spatial sign transformation.

4. PCA
We explored Principal component analysis (PCA) which is a linear dimensionality reduction
technique in order to find the best combination of predictors for modelling. This was employed
with the idea to see if PCA components perform well compared to transformed data. The
predictors were centered and scaled with YeoJohnson transformation applied, after which PCA
was performed on these predictors. PCA needed 42 components to capture 95% variance in the
data. Figure 10 shows the cumulative variance explained.

![image](https://github.com/user-attachments/assets/71e983c7-b96f-4cc9-a3e1-799f9831f6de)


5. Splitting of the Data
The response variable is continuous, i.e., price. The data is split randomly into an 80/20 ratio
with 80% of the data being used for training and 20% for testing. The train dataset consists of
4811 samples and the test set consists of 1200 samples. A 10-fold cross validation resampling
method is utilized. Given this is a regression problem, RMSE was used as the metric to choose
the best parameters.
6. Model Fitting
A range of linear and non-linear regression models are used to train on the training data. The
models are tested on the testing data with RMSE as our primary metric along with R
2
to evaluate
the model. The models are tested with multiple hyperparameters which will ensure the data is
11
tested on different combinations to avoid underfitting or overfitting. The model should be able to
generalize well on the unseen test data to obtain the best evaluation metrics. The table 2
showcases how the linear and non-linear regression models performed on the Airbnb Seattle
dataset.

![image](https://github.com/user-attachments/assets/756d1c99-76c1-4cab-ab7e-fc51a7fbf60d)

The same models were now trained and tested with PCA data to see if there is any development
in the performance of the models. The summary in table 3 shows how well PCA components
have performed.

![image](https://github.com/user-attachments/assets/5a978492-56ae-47e1-a5f4-2cd68bee219e)

The best model overall is Neural Networks with the lowest training RMSE of 0.2857756 and
the highest R2 score of 0.6591239. The testing RMSE is 0.2292414 and the R2 metric is
0.6691850. Neural Networks perform the best among all the models throughout all the training
13
and testing RMSE and R2 evaluation metrics. The PCA version of SVM performed close to
neural networks, making it the second best model. Also, the PCA version and the transformed
data performed closely similar, yet, the transformed data performed better overall.
Table 4 shows the performance of the best models on the training and testing sets. With testing
RMSE of 0.22 Neural Networks performs the best while explaining 0.66 variance in the data.
The PCA version of SVM closely follows neural networks with 0.24 testing RMSE and 0.63
Rsquared.

![image](https://github.com/user-attachments/assets/07d3384c-0998-4310-a542-0f894ecdacb9)

The most important predictors for the best model are given below:
only 20 most important variables shown (out of 51)
Overall
beds 100.000
bathrooms 73.010
bedrooms.1 69.488
room_typePrivate.room 61.933
bedrooms.3 40.595
minimum_nights 14.548
bedrooms.2 13.263
availability_30 10.688
latitude 10.410
instant_bookable 8.723
host_is_superhost 6.937
host_acceptance_rate 6.204
first_review_year 5.693
first_review_month.7 4.993
last_review_year 4.617
maximum_nights 4.320
number_of_reviews 3.550
last_review_month.6 3.483
reviews_per_month 3.233
first_review_month.8 3.215

![image](https://github.com/user-attachments/assets/0f0c3cba-a2d3-4d6e-b04d-05856c626745)

7. Summary
The best model is Neural Networks with the hyperparameters of size = 10 , decay = 0.1 , bag =
FALSE to predict the price of an Airbnb homestay in Seattle. The RMSE for the model on the
test dataset is 0.2292414 and an R2 of 0.6691850. The RMSE score of the model is low which
indicates that model predictions are remarkably close to the ground truth. The R2 score also
implies that the model is not effectively capturing the variance of the response variable which
indicates a moderate fit on the training data. The conclusion is that the Seattle Airbnb dataset
does not contain good enough predictors for predicting the response variable price. Transforming
the response model was the key to achieving the impressive RMSE scores displayed by the
models.





