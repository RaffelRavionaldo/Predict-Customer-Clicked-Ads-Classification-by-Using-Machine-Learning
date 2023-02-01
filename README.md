# Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning
The forth Mini project from rakamin academy

To run this code, dont forget to install requirement.
thanks to rakamin that give me this task, specially Mr. Abdullah Ghifari as my mentor and Mr. Travis Tang for the article about LazyPredict, reader can see the article via this link : https://pub.towardsai.net/lazypredict-run-all-sklearn-algorithms-with-a-line-of-code-29d73d82499c

Reader who want see my presentation you can see the pdf file. but if you dont want it just read below.

## Customer Type and Behaviour Analysis on Advertisement
### Univariate Analysis

![image](https://user-images.githubusercontent.com/94748637/216089717-0f88f877-7d89-4d11-8c67-fe02b3b65cb9.png)

From the graph, we know that customer tend to click our ad if they are :
1. Have <= 60 daily time spent on our site.
2. Are >=40 years old. 
3. Have an income <= 3.2 hundred million
4. And have <= 170 daily internet usage.

![image](https://user-images.githubusercontent.com/94748637/216089890-a4698a4f-ed8b-492a-88b4-6d971d2ddc9e.png)

From the graph, we know that woman have a percentage above 50% to click on our ads, and customer is more interested with our ads about house, finance, fashion and automotive,this can be said because of the percentage customer click our ads above 50%.

### Bivariate Analysis

![image](https://user-images.githubusercontent.com/94748637/216090775-8b868682-d840-4bd4-b91d-2bba3d8d7737.png)

From the graph, we know :
1. The older our customer, have a few daily time spent on site and daily internet usage they tend to click our ads.
2. The younger our customer, have a more daily time spent on site and daily internet usage they tend to not click our ads.

### Multivariate Analysis

![image](https://user-images.githubusercontent.com/94748637/216091294-23efe51e-3d01-48b1-9178-00424345a489.png)

From graph, we know that Daily Internet Usage and Daily Time spent on site have a highest correlation between another column, the second highest correlation is Age with Daily Internet Usage and the third is area income with daily internet usage.

![image](https://user-images.githubusercontent.com/94748637/216091468-a5145931-f393-4204-bd90-636f53c524b7.png)

Using Chi-Square for know association from category column, we know that clicked on Ad have a high correlation with city and  Male with category.

## Data Cleaning & Preprocessing.
### Missing Value
![image](https://user-images.githubusercontent.com/94748637/216092880-7ff1e3ce-0c96-497b-8d31-951bd82f201b.png)

we have 4 columns that have null data, 3 of then is numeric and others is categories.

![image](https://user-images.githubusercontent.com/94748637/216093098-6785e26e-0fa2-4540-b342-62535e9fbd94.png)

to fill null data, I use median for numeric columns and mode for categories column.

### Feature Encoding
#### Extract Datetime data

![image](https://user-images.githubusercontent.com/94748637/216093373-ac13c9e2-9aa9-47ee-ae2c-2f84202f30d4.png)

We get 5 new column, there are year, month, day, weekday and is_weekend, dont forget to remove timestamp column because that tabel cant be used to train ML model.

before we do feature transformation at categories columns, we divide them to 2 depends on their unique value.

![image](https://user-images.githubusercontent.com/94748637/216093864-2fbac63f-7b6e-42a5-8f3d-7a5704cb8a40.png)

#### Label Encoding
Is used For Category column that have 2 unique value or ordinal data.

![image](https://user-images.githubusercontent.com/94748637/216094021-16b3af9f-dfc0-4aab-9ceb-104cce1c6717.png)

#### One Hot Encoding
Is used for category column that have >2 unique value or nominal data.

![image](https://user-images.githubusercontent.com/94748637/216094191-94d2335d-8c17-4efd-8d62-15d27eff2714.png)

#### Scalling 
we do it for numeric column, some ML model will have a better accuracy if scale from every numeric column is same.
before do this, we need make a list that contains numeric columns.

![image](https://user-images.githubusercontent.com/94748637/216094609-ac5d2cc2-ea31-425e-ae33-3b8b78112d54.png)

#### Split data
we split the data into 2 parts, there are feature and target.

![image](https://user-images.githubusercontent.com/94748637/216094874-d63b6cfe-f648-43a6-b1fa-c39de04a3caa.png)

## Modelling
For this modelling, we will do some experiment, there are :

    1. First model will be trained by data that numeric columns dont do a scaling
    2. Second model will be trained by data that has passed all data preprocessing
    
And in this case, I will try library called Lazypredict to make a model, and another will be i try are LightGBM, RandomForest and XGBoost.

### Result of Experiment 1 

![image](https://user-images.githubusercontent.com/94748637/216095073-3bcb54d9-63c7-4ec3-a651-4e8dc627a47b.png)

The top model is XGBClassifier from XGBoost model, there have high evalution score (like 96% accuracy), but the time to predict testing data is 0.14 (see time taken column), is 7x longer then logistic regression that have 94% accuracy.

So if you need model that have higher accuracy, you can use XGBClassifier but if you need model that have a faster time to predict you can use Logistic regression.

### Result of Experiment 2

![image](https://user-images.githubusercontent.com/94748637/216095210-50bb95c5-2c26-421c-bc7d-0cbc21714847.png)

Compared with the result of experiment 1, there are no significant differences, just the time to predict test data is faster. We can look at XGBClassifier, at first experiment time taken is 0.14 second and at second experiment time taken is 0.09 second.

### Summary from the experiment

1. The model is trained by preprocessing data to get better result.
2. XGBClassifier from XGBoost get the better evaluation score then another model. But if you need model that can predict data faster than XGBoost I recommend to use logistic regression.
3. Because I need model that have a better evaluation score, I will choose XGBClassifier.34-8e24-c3eabbe0dd84.png)

### Feature Importance

![image](https://user-images.githubusercontent.com/94748637/216095597-9a60f0e0-5139-4d08-8746-8afdd338f001.png)

There are top 4 features that affect customers whether they click on our ads or not, namely Area income, Daily Internet Usage, Daily Time Spent on Site and Age.

## Business Recommendation
Based on Anaysis and feature importance from model, it can be concluded that :

1. We need to increase showing our ads to customer that meet the following requirement : They have income maximum 3.2 hundred million, are >=40 years old, have <= 60 minutes daily time spent on our site and  have <= 170 minutes daily internet usage.

2. For customer that don’t meet criteria at number 1, we need to decrease showing our ads because they are have low amount customer to click our ads. So that we can maximize our budget in advertising.

### Model Based - Simulation

We have a balanced amount of data between targets (50% click our ads and 50% no click our ads). Let’s count if we don’t use our model/do business recommendation :
Assumption : 
We show our ads at Google searchs Ads that have average CPM $38.40 (at 2021 via topdraw.com), let’s say if customer click our ads we got $0.1. so :
Cost : $38.40
Revenue = (1000 * 50%) * $0.1 = $50
Profit = Revenue – Cost = $50 - $38.40 = $11.6

** read about CPM at tbis link : https://www.investopedia.com/terms/c/cpm.asp
** topdraw full link = https://www.topdraw.com/insights/is-online-advertising-expensive/

Now if we use a ML model that has 96% accuracy to determine whether our customers will see our ads or not. We can get a profit per 1000 views of (We use the same assumption as before) :

Cost : $38.40
Revenue = (1000 * 96%) * $0.1 = $96
Profit = Revenue – Cost = $96 - $38.40 = $57.6

Is nearly 5 times bigger than if we don’t use ML model.
