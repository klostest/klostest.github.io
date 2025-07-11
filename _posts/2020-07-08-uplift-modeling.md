---
layout: post
title:  "Supercharging customer touchpoints with uplift modeling"
date:   2020-07-08
permalink: /uplift-modeling/
post_description: An introduction to a powerful way of predicting individual treatment effects, with synthetic data in Python using pandas and XGBoost.
post_image: "/assets/images/2020-07-08-uplift-modeling/space-shuttle-774_1280.jpg"
category: Technical coding
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<img src="/assets/images/2020-07-08-uplift-modeling/space-shuttle-774_1280.jpg" alt="Lift off, from https://pixabay.com/photos/space-shuttle-rocket-lift-off-774/" style="width: 500px; border: 10px solid white;"/>

Image from [Pixabay](https://pixabay.com/photos/space-shuttle-rocket-lift-off-774/)

In this post I will introduce the concept of uplift modeling and make a case for why it's an important part of the data scientist's toolbox of methods to increase business value. Then I'll show a simple way to build an uplift model and demonstrate a few uplift model evaluation metrics, using synthetic data in Python.

This post is available as a Jupyter notebook <a href="https://github.com/klostest/uplift_modeling/blob/master/uplift-modeling.ipynb" target="_blank">here on Github</a>.

- [Introduction](#Intro)
- [Data for uplift modeling: experiments are key](#Experiments)
- [Mechanics of the model](#Mechanics)
- [Example dataset](#Example)
- [Analyze experimental results](#Analyze)
- [Build an uplift model](#Build)
- [Model evaluation](#Model-evaluation)
    - [Quantile metrics](#quantile-metrics)
        - [Get score quantiles](#get-quantiles)
        - [Uplift quantile chart](#quantile-chart)
        - [Calibration of uplift](#calibration)
    - [Cumulative metrics](#cumulative-metrics)
        - [Cumulative gain chart](#Cumulative-gain)
        - [Cumulative gain curve](#gain-curve)
- [Conclusion](#conclusion)
- [References](#references)

# Introduction <a class="anchor" id="Intro"></a>

Of the myriad ways that machine learning can create value for businesses, uplift modeling is one of the lesser known, compared to methods such as supervised classification or regression. But for many use cases, it may be the most effective modeling technique. In any situation where there is a costly action a business can selectively take for different customers, in hopes of influencing their behavior, uplift modeling should be a strong candidate for determining strategy. This is because uplift modeling aims to find the subset of customers that would be most influenced by the action. Identifying this segment is important for maximizing the return on investment in a business strategy.

For example, in offering a coupon, a business is taking a potential revenue hit: if a customer buys and uses the coupon, revenue will be decreased by the value of the coupon. But, if the coupon persuades the customer to make a purchase, when they otherwise wouldn’t have, then it may still be worth it. These kinds of customers are called “persuadables” in the terminology of uplift modeling, which breaks things down into customer behavior with and without “treatment”, where treatment in this example is receiving a coupon.

<img src="https://storage.googleapis.com/wf-blogs-engineering-media/2018/10/e45e2d97-confmatrix_alt.png" alt="Drawing" style="width: 400px;"/>

Image from Yi and Frost (2018)

The goal of uplift modeling, also known as net lift or incremental response modeling, is to identify the "persuadables", not waste efforts on "sure things" and "lost causes", and avoid bothering "sleeping dogs", or those who would react negatively to the treatment, if they exist. Uplift modeling has found application in many domains including marketing, the classic use case illustrated here, as well as <a href="https://blogs.oracle.com/datascience/data-driven-debt-collection-using-machine-learning-and-predictive-analytics" target="_blank">debt collection</a> and <a href="https://go.forrester.com/blogs/13-06-27-how_the_obama_campaign_used_predictive_analytics_to_influence_voters/" target="_blank">political campaigns</a>.

# Data for uplift modeling: experiments are key <a class="anchor" id="Experiments"></a>
Now that we know the goal of uplift modeling, how do we get there? A typical starting point for building an uplift model is a dataset from a randomized, controlled experiment: we need a representative sample of all different kinds of customers in both a treatment group, as well as a control group that didn't receive treatment. If the proportion of customers making a purchase is significantly higher in the treatment group than the control group, we know that the promotion is “working” in the sense that it encourages a purchase on average across all customers. This is called the average treatment effect (ATE). Quantifying the ATE is the typical outcome of an A/B test.

However, it may be that only a portion of customers within the treatment group are responsible for most of the ATE we observe. As an extreme example, maybe half of the customers in the treatment group were responsible for the entire ATE, whereas the promotion had no effect on the other half. If we had some way to identify the persuadable segment of customers ahead of time, who would more readily respond to treatment, then we would be able to concentrate our resources on them, and not waste time on those for whom the treatment would have little or no effect. We may need to find other promotions to engage the non-responders. In the process of determining variable treatment effects from person to person, conditional on the different traits these people have, we’re looking for the individual treatment effect (ITE), also called the conditional average treatment effect (CATE). This is where machine learning and predictive modeling come into the picture.

# Mechanics of the model <a class="anchor" id="Mechanics"></a>
A classic technique for structuring an uplift model is to predict, for an individual customer, their likelihood of purchase if they are treated, and also the likelihood if they are not treated. These two probabilities are then subtracted to obtain the uplift: how much more likely is a purchase if treatment is given? This can be accomplished in two ways, where in both cases the binary response variable of the model is whether or not the customer made a purchase after the treatment:
- Lumping the treatment and control groups together into one data set and training a single model where treatment is a binary feature. In the inference phase, the model is used to make two predictions for each instance, the first with treatment = 1 and the second with treatment = 0. This is called the "S-Learner" approach since it uses a Single model.
- Training separate models for the treatment and control groups. In the inference phase, the treatment and control models are both used to obtain predictions for each instance. This is called the "T-Learner" approach since it uses Two models.

The two approaches are summarized in the following schematic:

<img src="/assets/images/2020-07-08-uplift-modeling/s-t-learner.svg" alt="S- and T-Learner differences" style="width: 900px;"/>

These approaches are widely documented in the literature on uplift modeling and causal inference (Lee et al. 2013, Gutierrez and Gerardy 2016). They have the advantage of being relatively simple and intuitive, and can be implemented using binary classification modeling techniques that many data scientists are familiar with, as well as specialized packages in enterprise software such as SAS (Lee et al. 2013). At the same time, causal inference is an active area of research within machine learning and other approaches may achieve better model performance. Different approaches include tree-based models designed to target uplift (reviewed in Gutierrez and Gerardy 2016), target variable transformation (Yi and Frost 2018), and other more recent innovations such as the X-Learner (Kunzel et al. 2019).

In all varieties, uplift modeling faces a fundamental challenge. The goal is to predict, for an individual customer, their likelihood of purchase if treated, and also the likelihood if not treated, to calculate the uplift. But in reality, we never observe the outcome for someone who was both treated and not treated, because this is impossible! Someone is either treated, or not. In mathematical modeling, it’s typically a problem if we can’t observe all the outcomes we’re interested in. This challenge illustrates the counterfactual nature of uplift modeling, and the importance of randomized experiments to understand the CATE across all types of customers.

<img src="/assets/images/2020-07-08-uplift-modeling/fork-2115485_960_720.jpg" alt="The choice of treating or not treating, from https://pixabay.com/images/id-2115485/" style="width: 400px;"/>

Image from [Pixabay](https://pixabay.com/images/id-2115485/)

Gutierrez and Gerardy (2016) summarize this challenge and point the way forward:

>Estimating customer uplift is both a Causal Inference and a Machine Learning problem. It is a causal inference problem because one needs to estimate the difference between two outcomes that are mutually exclusive for an individual (either person *i* receives a promotional e-mail or does not receive it). To overcome this counter-factual nature, uplift modeling crucially relies on randomized experiments, i.e. the random assignment of customers to either receive the treatment (the treatment group) or not (the control group). Uplift modeling is also a machine learning problem as one needs to train different models and select the one that yields the most reliable uplift prediction according to some performance metrics. This requires sensible cross-validation strategies along with potential feature engineering.

Let's explore these concepts using an example dataset, by building an S-Learner model and evaluating it.


```python
# load packages
import numpy as np
import pandas as pd

from statsmodels.stats.proportion import proportions_ztest

import sklearn as sk
from sklearn.metrics import auc
import xgboost as xgb

import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

import pickle
```

# Example dataset <a class="anchor" id="Example"></a>
The most straightforward way to build an uplift model is to start with data from a randomized controlled experiment. This way, both the treatment and control groups should have a representative sample of the population of customers. Outside of designed experiments, quasi-experimental data may be available if a natural control group exists as part of a business's normal operations. Treatment and control groups can also be approximated by a technique known as propensity score matching, available in the CausalML package that also offers a suite of uplift modeling tools (CausalML).

Here we use synthetic data from a recent publication (Zhao et al. 2020), which are publicly available [here](https://zenodo.org/record/3653141#.XwEVz5NKjYU). These data simulate a designed experiment with an even split between treatment and control groups. We load only the first 10,000 rows from this dataset, which is the first of "100 trials (replicates with different random seeds)". The dataset is constructed so that some features are predictive of the outcome, some are uninformative, and some are predictive of the treatment effect specifically.

The columns we're interested in are `treatment_group_key`, which identifies whether or not the customer received treatment, `conversion` which is 1 if the customer made a purchase and 0 if not, and the 36 synthetic features which all start with `x`. In real data, the features may correspond to such things as customer purchase history, demographics, and other quantities a data scientist may engineer with the hypothesis that they would be useful in modeling uplift.

Let's load the data and briefly explore it.


```python
df = pd.read_csv('data/uplift_synthetic_data_100trials.csv', nrows=10000)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 43 columns):
     #   Column                      Non-Null Count  Dtype  
    ---  ------                      --------------  -----  
     0   Unnamed: 0                  10000 non-null  int64  
     1   trial_id                    10000 non-null  int64  
     2   treatment_group_key         10000 non-null  object 
     3   conversion                  10000 non-null  int64  
     4   control_conversion_prob     10000 non-null  float64
     5   treatment1_conversion_prob  10000 non-null  float64
     6   treatment1_true_effect      10000 non-null  float64
     7   x1_informative              10000 non-null  float64
     8   x2_informative              10000 non-null  float64
     9   x3_informative              10000 non-null  float64
     ...
     42  x36_uplift_increase         10000 non-null  float64
    dtypes: float64(39), int64(3), object(1)
    memory usage: 3.3+ MB



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>trial_id</th>
      <th>treatment_group_key</th>
      <th>conversion</th>
      <th>control_conversion_prob</th>
      <th>treatment1_conversion_prob</th>
      <th>treatment1_true_effect</th>
      <th>x1_informative</th>
      <th>x2_informative</th>
      <th>x3_informative</th>
      <th>...</th>
      <th>x27_irrelevant</th>
      <th>x28_irrelevant</th>
      <th>x29_irrelevant</th>
      <th>x30_irrelevant</th>
      <th>x31_uplift_increase</th>
      <th>x32_uplift_increase</th>
      <th>x33_uplift_increase</th>
      <th>x34_uplift_increase</th>
      <th>x35_uplift_increase</th>
      <th>x36_uplift_increase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>control</td>
      <td>1</td>
      <td>0.516606</td>
      <td>0.572609</td>
      <td>0.056002</td>
      <td>-1.926651</td>
      <td>1.233472</td>
      <td>-0.475120</td>
      <td>...</td>
      <td>-0.378145</td>
      <td>-0.110782</td>
      <td>1.087180</td>
      <td>-1.222069</td>
      <td>-0.279009</td>
      <td>1.013911</td>
      <td>-0.570859</td>
      <td>-1.158216</td>
      <td>-1.336279</td>
      <td>-0.708056</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>treatment1</td>
      <td>1</td>
      <td>0.304005</td>
      <td>0.736460</td>
      <td>0.432454</td>
      <td>0.904364</td>
      <td>0.868705</td>
      <td>-0.285977</td>
      <td>...</td>
      <td>-0.742847</td>
      <td>0.700239</td>
      <td>0.001867</td>
      <td>-0.069362</td>
      <td>0.045789</td>
      <td>1.364182</td>
      <td>-0.261643</td>
      <td>0.478074</td>
      <td>0.531477</td>
      <td>0.402723</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>treatment1</td>
      <td>0</td>
      <td>0.134277</td>
      <td>0.480985</td>
      <td>0.346709</td>
      <td>1.680978</td>
      <td>1.320889</td>
      <td>0.059273</td>
      <td>...</td>
      <td>0.748884</td>
      <td>-0.856898</td>
      <td>-0.268034</td>
      <td>-2.181874</td>
      <td>1.473214</td>
      <td>-1.256641</td>
      <td>0.901139</td>
      <td>2.029204</td>
      <td>-0.280445</td>
      <td>0.873970</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>treatment1</td>
      <td>1</td>
      <td>0.801968</td>
      <td>0.858532</td>
      <td>0.056563</td>
      <td>-0.335774</td>
      <td>-2.940232</td>
      <td>-0.302521</td>
      <td>...</td>
      <td>0.151074</td>
      <td>0.067547</td>
      <td>-0.839246</td>
      <td>0.587575</td>
      <td>0.412081</td>
      <td>0.141189</td>
      <td>0.369611</td>
      <td>-0.364984</td>
      <td>-1.509045</td>
      <td>-1.335023</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>control</td>
      <td>0</td>
      <td>0.063552</td>
      <td>0.060142</td>
      <td>-0.003410</td>
      <td>-0.475881</td>
      <td>-0.485793</td>
      <td>0.978582</td>
      <td>...</td>
      <td>-1.287117</td>
      <td>1.256396</td>
      <td>-1.155307</td>
      <td>-0.414787</td>
      <td>1.163851</td>
      <td>0.698114</td>
      <td>0.088157</td>
      <td>0.478717</td>
      <td>-0.680588</td>
      <td>-2.730850</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 43 columns</p>
</div>



Of these 10,000 records, how many are in the treatment group, and how many are in the control group?


```python
df['treatment_group_key'].value_counts()
```




    control       5000
    treatment1    5000
    Name: treatment_group_key, dtype: int64



There is a 50/50 split. Let's encode the treatment variable as a binary 0/1:


```python
df['treatment_group_key'] = df['treatment_group_key'].map(arg={'control':0, 'treatment1':1})
```

# Analyze experimental results <a class="anchor" id="Analyze"></a>

What was the overall conversion rate?


```python
df['conversion'].mean()
```




    0.3191



What's the conversion rate in the treatment group versus the control group?


```python
exp_results_df = \
df.groupby('treatment_group_key').agg({'conversion':['mean', 'sum', 'count']})
exp_results_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">conversion</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>sum</th>
      <th>count</th>
    </tr>
    <tr>
      <th>treatment_group_key</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.2670</td>
      <td>1335</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.3712</td>
      <td>1856</td>
      <td>5000</td>
    </tr>
  </tbody>
</table>
</div>




```python
(exp_results_df.loc[1,('conversion', 'mean')]
 - exp_results_df.loc[0,('conversion', 'mean')]).round(4)
```




    0.1042



There is a substantially higher conversion rate in the treatment group (37%) than the control group (27%), indicating the treatment is effective at encouraging conversion: the ATE is positive and is about 10%.

Often in real data the difference is not so large and a significance test is usually conducted to determine the result of the A/B test.


```python
proportions_ztest(count=exp_results_df[('conversion', 'sum')],
                  nobs=exp_results_df[('conversion', 'count')])
```




    (-11.177190529878043, 5.273302441543889e-29)



The p-value is the second quantity returned from the proportion test and is much smaller than 0.05, or pretty much any other threshold used to decide significance. So we know there's a significant ATE. This is the typical starting point for uplift modeling. If we were to have observed that the treatment did not increase conversion rate, while theoretically it may be possible to find a persuadable segment of customers using uplift modeling, in practice it may not be a worthwhile endeavor. This likely depends on the specific problem at hand.

However in our case, having observed a substantial treatment effect, we proceed to use uplift modeling to find the CATE, and see if we can identify those persuadables.

# Build an uplift model <a class="anchor" id="Build"></a>
Here I'll use an `XGBClassifier` to train an S-Learner; that is, a single model including all the features, where the treatment indicator is also a feature. I'll split the data into training and validation sets (80/20 split), for early stopping. I'll also use the validation set to illustrate model evaluation metrics. In a real project, a held out test set should be reserved from this process, where the evaluation metrics on the test set would be used for final evaluation of the model.


```python
train, valid = sk.model_selection.train_test_split(df, test_size=0.2,random_state=42)
```


```python
print(train.shape, valid.shape)
```

    (8000, 43) (2000, 43)


Specify the features as a list. This includes the treatment column and all features, which are the 8th column onward:


```python
features = ['treatment_group_key'] + df.columns.tolist()[7:]
```


```python
print(features)
```

    ['treatment_group_key', 'x1_informative', 'x2_informative', 'x3_informative', 'x4_informative', 'x5_informative', 'x6_informative', 'x7_informative', 'x8_informative', 'x9_informative', 'x10_informative', 'x11_irrelevant', 'x12_irrelevant', 'x13_irrelevant', 'x14_irrelevant', 'x15_irrelevant', 'x16_irrelevant', 'x17_irrelevant', 'x18_irrelevant', 'x19_irrelevant', 'x20_irrelevant', 'x21_irrelevant', 'x22_irrelevant', 'x23_irrelevant', 'x24_irrelevant', 'x25_irrelevant', 'x26_irrelevant', 'x27_irrelevant', 'x28_irrelevant', 'x29_irrelevant', 'x30_irrelevant', 'x31_uplift_increase', 'x32_uplift_increase', 'x33_uplift_increase', 'x34_uplift_increase', 'x35_uplift_increase', 'x36_uplift_increase']


Assemble the training and validation sets for training the XGBoost classifer:


```python
X_train = train[features]
y_train = train['conversion']
X_valid = valid[features]
y_valid = valid['conversion']
```


```python
eval_set = [(X_train, y_train), (X_valid, y_valid)]
```

Now it's time to instantiate and train the model.


```python
# Train an xgboost model
model = xgb.XGBClassifier(learning_rate = 0.1,
                          max_depth = 6,
                          min_child_weight = 100,
                          objective = 'binary:logistic',
                          seed = 42,
                          gamma = 0.1,
                          silent = True,
                          n_jobs=2)
```


```python
%%time
model.fit(X_train, y_train, eval_set=eval_set,\
          eval_metric="auc", verbose=True, early_stopping_rounds=30)
```

    [0]	validation_0-auc:0.693049	validation_1-auc:0.648941
    Multiple eval metrics have been passed: 'validation_1-auc' will be used for early stopping.
    
    Will train until validation_1-auc hasn't improved in 30 rounds.
    [1]	validation_0-auc:0.718238	validation_1-auc:0.656877
    [2]	validation_0-auc:0.72416	validation_1-auc:0.667244
    [3]	validation_0-auc:0.727643	validation_1-auc:0.669992
    ...
    [99]	validation_0-auc:0.852237	validation_1-auc:0.762969
    CPU times: user 6.7 s, sys: 87.8 ms, total: 6.79 s
    Wall time: 3.48 s





    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0.1,
                  learning_rate=0.1, max_delta_step=0, max_depth=6,
                  min_child_weight=100, missing=None, n_estimators=100, n_jobs=2,
                  nthread=None, objective='binary:logistic', random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=42,
                  silent=True, subsample=1, verbosity=1)



The training process completes and we can see the model has a fairly high validation AUC. The training AUC is even higher than this, meaning technically the model is overfit. Normally I'd do a hyperparameter search, but I found the values used here to provide sensible results for the purpose of illustrating uplift modeling with this dataset.

As a practical side note, I've found that in some cases, when using a T-Learner (not shown here), that overfitting to the training set can cause unexpected results when calculating uplift. In my experience the issue could be remedied by decreasing `max_depth` or increasing `min_child_weight` in the `XGBClassifier`, in other words decreasing the amount of overfitting.

Another point to consider in model building is feature selection, which I've omitted here. In the context of uplift modeling, one could use the uplift model evaluation metrics introduced below on a validation set as a way to select features, for example by recursive feature elimination. Feature selection for uplift models is also the topic of recent research, including the paper that is the source of the dataset used here (Zhao et al. 2020).

# Model evaluation <a class="anchor" id="Model-evaluation"></a>
Now, we have our uplift model. The model building for an S-Learner is pretty simple, if you are already familiar with binary classification. To actually calculate the uplift for a given data set, with this approach we need to score the model twice, once with treatment = 1 and again with treatment = 0, then subtract these to get the uplift. Here we do this for the validation set, then plot a histogram of the uplift scores.


```python
X_valid_0 = X_valid.copy(); X_valid_0['treatment_group_key'] = 0
X_valid_1 = X_valid.copy(); X_valid_1['treatment_group_key'] = 1
Uplift = model.predict_proba(X_valid_1)[:,1]\
    - model.predict_proba(X_valid_0)[:,1]
```


```python
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['figure.figsize'] = (6,4)
plt.hist(Uplift, bins=50)
plt.xlabel('Uplift score')
plt.ylabel('Number of observations in validation set')
```




    Text(0, 0.5, 'Number of observations in validation set')




![png](/assets/images/2020-07-08-uplift-modeling/output_38_1.png)


The distribution of uplift is mostly positive, which makes sense since we know from our analysis of the experiment that the treatment encourages conversion on average. However some instances have negative uplift, meaning treatment actually __discourages__ conversion for some people. In other words there appears to be a sleeping dog effect in these data.

The main questions now are: should we trust these results? How do we know how good this model is? Metrics for uplift model evaluation are more complex than typical metrics used in supervised learning, such as the ROC AUC for classification tasks or RMSE for regression. Generally speaking, uplift evaluation metrics make a comparison of the conversion rate between the treatment and control groups, for different ranges of the predicted uplift score. For those with high uplift scores, we'd expect to see a larger difference between treatment and control, while those with lower uplift scores should have a smaller difference, or even a larger conversion rate in the control group in the case of sleeping dogs (i.e. negative difference).

## Quantile metrics <a class="anchor" id="quantile-metrics"></a>
A popular way to evaluate an uplift model is with a quantile chart. This will give a quick visual impression of whether the model is "working", in the sense of sloping true uplift. To create the quantile chart, we start with the uplift predictions for our validation set, and bin the instances into quantiles based on these scores. The number of quantiles depends on how much data we have, although 10 is a pretty typical number in practice (deciles). Then, within each bin, we'll find the difference in conversion rate for those who were in the treatment group and those who were in the control group. If the model is working well, we should see a larger positive difference in the highest decile, decreasing to a small or negative difference in the lowest decile (i.e. treatment rate similar to control rate, or lower than control rate). In other words, as predicted uplift increases, the true uplift from control to treatment group should increase as well.

### Get score quantiles <a class="anchor" id="get-quantiles"></a>
Create a new `DataFrame` from the validation data, to add the uplift scores and quantiles.


```python
Uplift.shape
```




    (2000,)




```python
valid.shape
```




    (2000, 43)




```python
valid_w_score = valid.copy()
valid_w_score['Uplift score'] = Uplift
```


```python
valid_w_score.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>trial_id</th>
      <th>treatment_group_key</th>
      <th>conversion</th>
      <th>control_conversion_prob</th>
      <th>treatment1_conversion_prob</th>
      <th>treatment1_true_effect</th>
      <th>x1_informative</th>
      <th>x2_informative</th>
      <th>x3_informative</th>
      <th>...</th>
      <th>x28_irrelevant</th>
      <th>x29_irrelevant</th>
      <th>x30_irrelevant</th>
      <th>x31_uplift_increase</th>
      <th>x32_uplift_increase</th>
      <th>x33_uplift_increase</th>
      <th>x34_uplift_increase</th>
      <th>x35_uplift_increase</th>
      <th>x36_uplift_increase</th>
      <th>Uplift score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6252</th>
      <td>6252</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.091233</td>
      <td>0.043904</td>
      <td>-0.047329</td>
      <td>0.128796</td>
      <td>0.582004</td>
      <td>2.027088</td>
      <td>...</td>
      <td>0.971923</td>
      <td>-1.332471</td>
      <td>0.218589</td>
      <td>0.700103</td>
      <td>0.074845</td>
      <td>0.580532</td>
      <td>-1.966506</td>
      <td>-0.322667</td>
      <td>-1.311465</td>
      <td>0.009574</td>
    </tr>
    <tr>
      <th>4684</th>
      <td>4684</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.101494</td>
      <td>0.069031</td>
      <td>-0.032464</td>
      <td>-0.249103</td>
      <td>-0.032255</td>
      <td>0.030591</td>
      <td>...</td>
      <td>0.015858</td>
      <td>-0.160939</td>
      <td>0.211841</td>
      <td>0.447512</td>
      <td>-0.269382</td>
      <td>-0.328579</td>
      <td>-0.297928</td>
      <td>-0.473154</td>
      <td>-1.592787</td>
      <td>-0.003638</td>
    </tr>
    <tr>
      <th>1731</th>
      <td>1731</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.408242</td>
      <td>0.816324</td>
      <td>0.408081</td>
      <td>0.694069</td>
      <td>-0.288068</td>
      <td>-0.588280</td>
      <td>...</td>
      <td>0.190106</td>
      <td>1.643789</td>
      <td>1.558527</td>
      <td>0.684027</td>
      <td>0.367873</td>
      <td>-0.744745</td>
      <td>0.378264</td>
      <td>0.532618</td>
      <td>0.103382</td>
      <td>0.288655</td>
    </tr>
    <tr>
      <th>4742</th>
      <td>4742</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.036061</td>
      <td>0.055446</td>
      <td>0.019384</td>
      <td>-1.012592</td>
      <td>0.239213</td>
      <td>-0.402686</td>
      <td>...</td>
      <td>-0.491389</td>
      <td>-0.947252</td>
      <td>-0.185026</td>
      <td>0.085944</td>
      <td>-0.661352</td>
      <td>-0.770008</td>
      <td>0.860812</td>
      <td>-0.749852</td>
      <td>-0.391099</td>
      <td>0.131196</td>
    </tr>
    <tr>
      <th>4521</th>
      <td>4521</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.206175</td>
      <td>0.125444</td>
      <td>-0.080731</td>
      <td>-1.564519</td>
      <td>-0.809688</td>
      <td>0.859528</td>
      <td>...</td>
      <td>1.053866</td>
      <td>1.378201</td>
      <td>-0.370168</td>
      <td>-0.690919</td>
      <td>0.383968</td>
      <td>0.745777</td>
      <td>0.693021</td>
      <td>-0.860461</td>
      <td>1.262036</td>
      <td>0.042706</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 44 columns</p>
</div>



Check that the treatment and control groups are approximately balanced, overall for the validation set (they should be since we used a random training/validation split but it's always good to check):


```python
valid_w_score['treatment_group_key'].value_counts()
```




    0    1011
    1     989
    Name: treatment_group_key, dtype: int64



Now, using the entire validation set (treatment and control groups together), make labels for the uplift score quantiles. We'll check that the treatment and control groups are balanced within the quantiles, since we'll be splitting the data by quantile and treatment to create the chart. Pandas has a convenient function to produce a series of labels according to which quantile an observation in the input series belongs to.


```python
score_quantiles, score_quantile_bins = \
pd.qcut(x=valid_w_score['Uplift score'],
        q=10,
        retbins=True,
        duplicates='drop')
```

From this function we get a column indicating which quantile each instance belongs to, represented by the bin edges:


```python
score_quantiles.head()
```




    6252    (-0.00339, 0.0186]
    4684    (-0.114, -0.00339]
    1731        (0.201, 0.398]
    4742        (0.121, 0.148]
    4521      (0.0391, 0.0548]
    Name: Uplift score, dtype: category
    Categories (10, interval[float64]): [(-0.114, -0.00339] < (-0.00339, 0.0186] < (0.0186, 0.0391] < (0.0391, 0.0548] ... (0.0941, 0.121] < (0.121, 0.148] < (0.148, 0.201] < (0.201, 0.398]]



We also get a list of all bin edges in `score_quantile_bins`, but we don't need it here. Now let's add the score quantile to the dataframe so we can use it for analysis.


```python
valid_w_score['Quantile bin'] = score_quantiles
valid_w_score[[
    'treatment_group_key', 'conversion', 'Uplift score', 'Quantile bin']].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>treatment_group_key</th>
      <th>conversion</th>
      <th>Uplift score</th>
      <th>Quantile bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6252</th>
      <td>0</td>
      <td>1</td>
      <td>0.009574</td>
      <td>(-0.00339, 0.0186]</td>
    </tr>
    <tr>
      <th>4684</th>
      <td>1</td>
      <td>0</td>
      <td>-0.003638</td>
      <td>(-0.114, -0.00339]</td>
    </tr>
    <tr>
      <th>1731</th>
      <td>1</td>
      <td>0</td>
      <td>0.288655</td>
      <td>(0.201, 0.398]</td>
    </tr>
    <tr>
      <th>4742</th>
      <td>1</td>
      <td>0</td>
      <td>0.131196</td>
      <td>(0.121, 0.148]</td>
    </tr>
    <tr>
      <th>4521</th>
      <td>0</td>
      <td>1</td>
      <td>0.042706</td>
      <td>(0.0391, 0.0548]</td>
    </tr>
    <tr>
      <th>6340</th>
      <td>0</td>
      <td>1</td>
      <td>0.113626</td>
      <td>(0.0941, 0.121]</td>
    </tr>
    <tr>
      <th>576</th>
      <td>1</td>
      <td>0</td>
      <td>-0.018831</td>
      <td>(-0.114, -0.00339]</td>
    </tr>
    <tr>
      <th>5202</th>
      <td>1</td>
      <td>0</td>
      <td>-0.036599</td>
      <td>(-0.114, -0.00339]</td>
    </tr>
    <tr>
      <th>6363</th>
      <td>1</td>
      <td>1</td>
      <td>0.113048</td>
      <td>(0.0941, 0.121]</td>
    </tr>
    <tr>
      <th>439</th>
      <td>0</td>
      <td>1</td>
      <td>0.074832</td>
      <td>(0.0726, 0.0941]</td>
    </tr>
  </tbody>
</table>
</div>



Check that the number of treated and control observations within quantile bins are similar, using groupby/count and some multiindex magic:


```python
count_by_quantile_and_treatment = valid_w_score.groupby(
    ['Quantile bin', 'treatment_group_key'])['treatment_group_key'].count()
count_by_quantile_and_treatment = count_by_quantile_and_treatment.unstack(-1)
count_by_quantile_and_treatment
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>treatment_group_key</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>Quantile bin</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(-0.114, -0.00339]</th>
      <td>93</td>
      <td>107</td>
    </tr>
    <tr>
      <th>(-0.00339, 0.0186]</th>
      <td>110</td>
      <td>90</td>
    </tr>
    <tr>
      <th>(0.0186, 0.0391]</th>
      <td>99</td>
      <td>101</td>
    </tr>
    <tr>
      <th>(0.0391, 0.0548]</th>
      <td>105</td>
      <td>95</td>
    </tr>
    <tr>
      <th>(0.0548, 0.0726]</th>
      <td>112</td>
      <td>88</td>
    </tr>
    <tr>
      <th>(0.0726, 0.0941]</th>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>(0.0941, 0.121]</th>
      <td>102</td>
      <td>98</td>
    </tr>
    <tr>
      <th>(0.121, 0.148]</th>
      <td>107</td>
      <td>93</td>
    </tr>
    <tr>
      <th>(0.148, 0.201]</th>
      <td>89</td>
      <td>111</td>
    </tr>
    <tr>
      <th>(0.201, 0.398]</th>
      <td>94</td>
      <td>106</td>
    </tr>
  </tbody>
</table>
</div>




```python
count_by_quantile_and_treatment.plot.barh()
plt.xlabel('Number of observations')
```




    Text(0.5, 0, 'Number of observations')




![png](/assets/images/2020-07-08-uplift-modeling/output_56_1.png)


Without being too precise about it, it doesn't appear that the score quantiles are unbalanced in terms of the proportion of treatment and control; they are similar in each bin. This is expected, as we are working with data from a randomized experiment, however again it's good to check such assumptions.

### Uplift quantile chart <a class="anchor" id="quantile-chart"></a>

On to the uplift quantile chart. We'll start by creating a mask we can use for the treatment group:


```python
validation_treatment_mask = valid_w_score['treatment_group_key'] == 1
```

Then we get the conversion rates within uplift score quantiles, separately for treatment and control groups:


```python
treatment_by_quantile = valid_w_score[validation_treatment_mask]\
    .groupby('Quantile bin')['conversion'].mean()
control_by_quantile = valid_w_score[~validation_treatment_mask]\
    .groupby('Quantile bin')['conversion'].mean()
```

Finally we calculate their difference, which is the true uplift within each score quantile.


```python
true_uplift_by_quantile = treatment_by_quantile - control_by_quantile
true_uplift_by_quantile.head(5)
```




    Quantile bin
    (-0.114, -0.00339]   -0.017486
    (-0.00339, 0.0186]    0.034343
    (0.0186, 0.0391]     -0.004600
    (0.0391, 0.0548]      0.021554
    (0.0548, 0.0726]      0.133929
    Name: conversion, dtype: float64



Now we have all the information needed to plot an uplift quantile chart.


```python
true_uplift_by_quantile.plot.barh()
plt.xlabel('True uplift')
```




    Text(0.5, 0, 'True uplift')




![png](/assets/images/2020-07-08-uplift-modeling/output_64_1.png)


The uplift quantile chart shows that, for the most part, true uplift increases from lower score bins to higher ones, which is what we'd expect to see if the model is working. So it appears our model can effectively segment out customers who more readily respond to treatment. In a real project, it would be important to repeat this analysis on a held-out test set, to confirm the model works with data that was not used at all for model training, since technically this validation set was used for early stopping in the model fitting process. However good performance on the validation set is a good sign and as long as the test set has similar characteristics to the training and validation sets, we'd expect to see similar performance there.

What can we learn from the quantile chart? From analysis of the experiment we know the ATE is about 10%. The quantile chart we created with the validation set tells us that by targeting the top decile of uplift scores, we may achieve a treatment effect of over 35%, a noticeable increase. The next few deciles appear to have larger treatment effects than the ATE as well. Depending on how expensive the treatment is to apply, it may make sense to target a limited portion of the population using this information.

We can also see that the sleeping dog effect has some support from observations of the true uplift. The bottom score decile, which consists entirely of negative scores, in fact has negative true uplift. So it appears that targeting the bottom 10% of the population, by uplift score, actually has a negative impact on the business.

### Calibration of uplift <a class="anchor" id="calibration"></a>

While the uplift quantile chart provides a qualitative snapshot telling us whether the model is effective at segmenting customers or not, we can go a step further in this direction and ask how accurate the model is at predicting uplift. This is the process of calibration, for which we'll need the average predicted uplift within the score quantiles:


```python
predicted_uplift_by_quantile = valid_w_score\
    .groupby(['Quantile bin'])['Uplift score'].mean()
predicted_uplift_by_quantile.head(5)
```




    Quantile bin
    (-0.114, -0.00339]   -0.035133
    (-0.00339, 0.0186]    0.008385
    (0.0186, 0.0391]      0.029582
    (0.0391, 0.0548]      0.047033
    (0.0548, 0.0726]      0.063634
    Name: Uplift score, dtype: float32



We'll put this together in a `DataFrame` with the true uplift that we calculated above, to create a scatter plot. If the uplift predictions are accurate, a scatter plot of predicted and true uplift should lie close to the one-one line.


```python
pred_true_uplift = pd.DataFrame({'Predicted Uplift':predicted_uplift_by_quantile,
                                 'True Uplift':true_uplift_by_quantile})

min_on_plot = pred_true_uplift.min().min()
max_on_plot = pred_true_uplift.max().max()

ax = plt.axes()
ax.plot([min_on_plot, max_on_plot], [min_on_plot, max_on_plot],
        linestyle='--', color='tab:orange', label='One-one line')
pred_true_uplift.plot.scatter(x='Predicted Uplift', y='True Uplift',
                              ax=ax, label='Model performance')

ax.legend()
```




    <matplotlib.legend.Legend at 0x1a1c84c990>




![png](/assets/images/2020-07-08-uplift-modeling/output_68_1.png)


Qualitatively, we can see from the calibration plot that mean predicted uplift is close to true uplift, by quantile. This calibration could be made more precise by calculating some sort of metric, perhaps MAE (mean absolute error), as a measure of model goodness of fit.

There are a few other ways these quantile-based analyses could be extended:
- The predictions for treatment = 1 and treatment = 0, that were used to calculate uplift, could be separately calibrated against the conversion rates by deciles of those scores, in the treatment and control groups respectively. This would be the calibration of predicted probability of conversion for these groups.
- Error bars could be included on all plots. For predicted probability of conversion, one could calculate the standard error of the mean within each bin of the treatment and control groups separately, while for true conversion rate one could use the normal approximation to the binomial. Then when subtracting the means of treatment and control to get uplift, the variances based on these calculations would be added, and the standard error of uplift could be calculated within each bin.
- In a business situation, the cost of treatment and the expected revenue of conversion should be known. The uplift quantile chart could be extended to represent uplift in revenue, which could be balanced against the cost of treatment to assess profitability of a business strategy.

## Cumulative metrics <a class="anchor" id="cumulative-metrics"></a>
### Cumulative gain chart <a class="anchor" id="Cumulative-gain"></a>
When using uplift scores in practice, a common approach is to rank customers in descending order, according to their uplift score. We can extend the quantile chart idea to calculate how many additional customers ("incremental customers") we can obtain by targeting a particular fraction of the population, in this way.

This idea underlies the cumulative gain chart. The formula for cumulative gain is given by Gutierrez and Gerardy (2016) as:

$$ \left( \frac{Y^T}{N^T} - \frac{Y^C}{N^C} \right) \left( N^T + N^C \right)$$

where $Y^T$ is the cumulative sum of conversions in each bin of the treatment group, starting with the highest score bin and proceeding down, and $N^T$ is the cumulative number of customers found in the same way; $Y^C$ and $N^C$ are similar cumulative sums for the control group. Cumulative gain effectively measures the cumulative uplift in probability of conversion, starting with the highest bin, and multiplies by the number of total customers in both treatment and control groups, to estimate the number of additional conversions that would occur if that number of customers were targeted.

To get the data for the cumulative gain chart, we will need to calculate the amount of customers in each score quantile bin, both in the treatment and control groups (we visualized this above but will recalculate it here) and also the sum of converted customers. Here we'll flip the result upside down with `.iloc[::-1]` to simulate the strategy of targeting the customers with highest uplift scores first, and proceeding down from there.


```python
treatment_count_by_quantile = valid_w_score[validation_treatment_mask]\
    .groupby('Quantile bin').agg({'conversion':['sum', 'count']}).iloc[::-1]

control_count_by_quantile = valid_w_score[~validation_treatment_mask]\
    .groupby('Quantile bin').agg({'conversion':['sum', 'count']}).iloc[::-1]
```

Here is how the treatment group looks, for example:


```python
treatment_count_by_quantile.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">conversion</th>
    </tr>
    <tr>
      <th></th>
      <th>sum</th>
      <th>count</th>
    </tr>
    <tr>
      <th>Quantile bin</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(0.201, 0.398]</th>
      <td>57</td>
      <td>106</td>
    </tr>
    <tr>
      <th>(0.148, 0.201]</th>
      <td>57</td>
      <td>111</td>
    </tr>
    <tr>
      <th>(0.121, 0.148]</th>
      <td>41</td>
      <td>93</td>
    </tr>
    <tr>
      <th>(0.0941, 0.121]</th>
      <td>38</td>
      <td>98</td>
    </tr>
    <tr>
      <th>(0.0726, 0.0941]</th>
      <td>37</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>



And the cumulative sums of conversions and total customers:


```python
treatment_count_by_quantile.cumsum().head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">conversion</th>
    </tr>
    <tr>
      <th></th>
      <th>sum</th>
      <th>count</th>
    </tr>
    <tr>
      <th>Quantile bin</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(0.201, 0.398]</th>
      <td>57</td>
      <td>106</td>
    </tr>
    <tr>
      <th>(0.148, 0.201]</th>
      <td>114</td>
      <td>217</td>
    </tr>
    <tr>
      <th>(0.121, 0.148]</th>
      <td>155</td>
      <td>310</td>
    </tr>
    <tr>
      <th>(0.0941, 0.121]</th>
      <td>193</td>
      <td>408</td>
    </tr>
    <tr>
      <th>(0.0726, 0.0941]</th>
      <td>230</td>
      <td>508</td>
    </tr>
  </tbody>
</table>
</div>



Putting all this together into the formula for cumulative gain shown above, we have:


```python
cumulative_gain = \
    ((treatment_count_by_quantile[('conversion','sum')].cumsum()
      / treatment_count_by_quantile[('conversion','count')].cumsum())
     -
     (control_count_by_quantile[('conversion','sum')].cumsum()
      / control_count_by_quantile[('conversion','count')].cumsum())) \
    * (treatment_count_by_quantile[('conversion','count')].cumsum()
       + control_count_by_quantile[('conversion','count')].cumsum())
```

We can now examine and plot the cumulative gain.


```python
cumulative_gain.round()
```




    Quantile bin
    (0.201, 0.398]         74.0
    (0.148, 0.201]        125.0
    (0.121, 0.148]        149.0
    (0.0941, 0.121]       172.0
    (0.0726, 0.0941]      209.0
    (0.0548, 0.0726]      237.0
    (0.0391, 0.0548]      242.0
    (0.0186, 0.0391]      240.0
    (-0.00339, 0.0186]    249.0
    (-0.114, -0.00339]    248.0
    dtype: float64




```python
cumulative_gain.plot.barh()
plt.xlabel('Cumulative gain in converted customers')
```




    Text(0.5, 0, 'Cumulative gain in converted customers')




![png](/assets/images/2020-07-08-uplift-modeling/output_80_1.png)


Cumulative gain gives another way to look at the potential impact of an uplift model-guided strategy. If we offer the treatment to every customer, we'll increase the number of converted customers by 248. However we can achieve a gain of 149 customers, about 60% of the maximum possible, by only offering treatment to the top 30% of customers (top 3 deciles), by uplift score. This is because as we move down the list, we're targeting customers with lower predicted individual treatment effect. The cumulative number of conversions may even go down from bin to bin, in the lower-valued bins of the chart, as we actually lose customers by targeting sleeping dogs.

### Cumulative gain curve <a class="anchor" id="gain-curve"></a>
The various charts above are all informative and intuitive ways to understand the performance of an uplift model, however they don't immediately lead to model performance metrics, by which different modeling approaches might be compared. In order to make this leap, the idea of cumulative gain can be generalized to a curve, in a similar way to how the receiver operating characteristic (ROC) curve is used to evaluate binary classifiers. To get an ROC curve, true positive rates and false positive rates are calculated as the threshold for positive classification is successively raised to include more and more instances in a data set, until all are included. By comparison, a cumulative gain curve measures the cumulative gain in conversions, as defined above, in the targeted population as more and more people are targeted according to descending uplift score, until all are targeted.

The gain curve is defined as

$$ f(t) = \left( \frac{Y_t^T}{N_t^T} - \frac{Y_t^C}{N_t^C} \right) \left( N_t^T + N_t^C \right)$$

where $t$ is the index of the customer, starting from the highest uplift score and proceeding down, and the other variables are defined similarly to the previous equation.

The gain curve calculation is available as part of the CausalML package, where it is called the uplift curve (CausalML). It can also be calculated fairly quickly in pandas. The first step is sorting on uplift score:


```python
sorted_valid = valid_w_score.sort_values('Uplift score', ascending=False)\
.reset_index(drop=True)

sorted_valid[['treatment_group_key', 'conversion', 'Uplift score']].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>treatment_group_key</th>
      <th>conversion</th>
      <th>Uplift score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0.397679</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0.375838</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0.372153</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0.364114</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>0.361814</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0.356630</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0.349603</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>1</td>
      <td>0.348355</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>1</td>
      <td>0.345382</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>0.340867</td>
    </tr>
  </tbody>
</table>
</div>



On to this `DataFrame`, we add a few columns which will make the calculation of the curve easier, following the notation in the equations above.


```python
sorted_valid['Y_T'] = \
    (sorted_valid['conversion'] * sorted_valid['treatment_group_key']).cumsum()
sorted_valid['Y_C'] = \
    (sorted_valid['conversion'] * (sorted_valid['treatment_group_key']==0)).cumsum()
sorted_valid['N_T'] = sorted_valid['treatment_group_key'].cumsum()
sorted_valid['N_C'] = (sorted_valid['treatment_group_key']==0).cumsum()
```


```python
sorted_valid[['treatment_group_key', 'conversion', 'Uplift score',
              'Y_T', 'Y_C', 'N_T', 'N_C']].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>treatment_group_key</th>
      <th>conversion</th>
      <th>Uplift score</th>
      <th>Y_T</th>
      <th>Y_C</th>
      <th>N_T</th>
      <th>N_C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0.397679</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0.375838</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0.372153</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0.364114</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>0.361814</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0.356630</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0.349603</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>1</td>
      <td>0.348355</td>
      <td>3</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>1</td>
      <td>0.345382</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>0.340867</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



Now the calculation of the gain curve can be done as follows:


```python
sorted_valid['Gain curve'] = (
    (sorted_valid['Y_T']/sorted_valid['N_T'])
    -
    (sorted_valid['Y_C']/sorted_valid['N_C'])
    ) * (sorted_valid['N_T'] + sorted_valid['N_C'])
```

Let's examine the gain curve.


```python
sorted_valid['Gain curve'].plot()
plt.xlabel('Number of customers treated')
plt.ylabel('Gain in conversion')
```




    Text(0, 0.5, 'Gain in conversion')




![png](/assets/images/2020-07-08-uplift-modeling/output_90_1.png)


The gain curve looks fairly similar to how the gain chart above would look (if it were turned on its side), just more continuous, and largely tells the same story. One advantage of the curve, however, is that similar to the ROC curve, we can calculate an area under the curve, with the interpretation that larger areas mean a more performant model: we would like to be able to gain as many customers as possible, by targeting a few as possible. If we had perfect knowledge of who would respond positively to treatment, we would treat only those who would, and the gain curve as plotted above would have a slope of one initially, before leveling off and potentially declining if there were sleeping dogs. This would lead to a gain curve with a steep initial slope, that stayed high as long as possible, resulting in a large area under the curve.

Before calculating an AUC, it can be beneficial to normalize the data. As shown, the gain curve has units of customers on both the x- and y-axes. This can be good for visualizing things in terms of real-world quantities. However, if we wanted to assess performance on validation and test sets, for example, the areas under these curves may not be comparable as these datasets may have different numbers of observations. We can remedy this by scaling the curve so that both the x- and y-axes have a maximum of 1.

The scaled x-axis represents the fraction of the population targeted:


```python
gain_x = sorted_valid.index.values + 1
gain_x = gain_x/gain_x.max()
print(gain_x[:3])
print(gain_x[-3:])
```

    [0.0005 0.001  0.0015]
    [0.999  0.9995 1.    ]


And the scaled y-axis is the fraction of gain from treating the entire population:


```python
gain_y = (
    sorted_valid['Gain curve']
    /
    sorted_valid['Gain curve'].tail(1).values
    ).values

print(gain_y[:3])
print(gain_y[-3:])
```

    [nan  0.  0.]
    [1.00802087 1.00534686 1.        ]


Note the first entry in the normalized gain curve is `NaN`; there will always be at least one of these at the beginning because either $N^T_t$ or $N^C_t$ will be zero for at least the first observation, leading to a divide by zero error. So we'll drop entries from both the x and y vectors here to get rid of `NaN`s, which shouldn't be an issue if the data set is large enough.


```python
nan_mask = np.isnan(gain_y)
gain_x = gain_x[~nan_mask]
gain_y = gain_y[~nan_mask]
print(gain_y[:3])
print(gain_y[-3:])
```

    [0.         0.         0.00805023]
    [1.00802087 1.00534686 1.        ]


Now we can plot the normalized gain curve, along with a computed AUC. To this, we'll add a one-one line. Similar to the interpretation of a one-one line on an ROC curve, here this corresponds to the gain curve we'd theoretically expect by treating customers at random: the fraction of the gain we would get by treating all customers increases according to the fraction treated and the ATE.


```python
mpl.rcParams['font.size'] = 8
gain_auc = auc(gain_x, gain_y)

ax = plt.axes()
ax.plot(gain_x, gain_y,
        label='Normalized gain curve, AUC {}'.format(gain_auc.round(2)))
ax.plot([0, 1], [0, 1],
        '--', color='tab:orange',
        label='Random treatment')
ax.set_aspect('equal')
ax.legend()
ax.set_xlabel('Fraction of population treated')
ax.set_ylabel('Fraction of gain in converted customers')
ax.grid()
```


![png](/assets/images/2020-07-08-uplift-modeling/output_98_0.png)


Notice that, unlike an ROC curve, the gain curve can actually exceed 1.0 on the y-axis. This is because we may be able to gain more customers than the number we'd get by treating everyone, if we can avoid some sleeping dogs.

The AUC calculated here gives a general way to compare uplift model performance across different models and data sets, such as the training, validation, and testing sets for a given application. Normalized gain curves can be plotted together and have their AUCs compared in the same way ROC AUCs are compared for supervised classification problems. The interested reader may wish to develop a T-Learner model and compare with the S-Learner results shown here as an exercise.

As a final step here we'll save the validation set and the trained model, in case we want to do further analysis.


```python
# with open('data/validation_set_and_model_7_8_20.pkl', 'wb') as fname:
#     pickle.dump([valid_w_score, model], fname)
```

# Conclusion <a class="anchor" id="conclusion"></a>
The goal of uplift modeling is to create predictive models of the individual treatment effect. Such models allow data scientists to segment populations into groups that are more likely to respond to treatment, and those that are less so. With this goal, a variety of modeling techniques have been developed; uplift modeling continues to receive active research interest. The evaluation of uplift models is not as straightforward as that of supervised classification or regression models because it requires separate consideration, and comparison, of treatment and control groups. However, open source Python packages (CausalML, Pylift) have been created to facilitate uplift model development and evaluation. Several useful uplift evaluation techniques, some of which are available in those packages, were demonstrated here using Python and pandas.

---

Thanks to Pierre Gutierrez and Robert Yi for your input and feedback

# References <a class="anchor" id="references"></a>

[CausalML: a Python package that provides a suite of uplift modeling and causal inference methods using machine learning algorithms based on recent research. Accessed 7/5/2020.](https://causalml.readthedocs.io/en/latest/about.html)

[Gutierrez, Pierre and Jean-Yves Gerardy, 2016. Causal Inference and Uplift Modeling: A review of the literature. JMLR: Workshop and Conference Proceedings 67:1-13.](http://proceedings.mlr.press/v67/gutierrez17a/gutierrez17a.pdf)

[Kunzel, Soren R. et al., 2019. Metalearners for estimating heterogeneous treatment effects using maching learning. PNAS March 5, 2019 115 (10) 4156-4165](https://www.pnas.org/content/116/10/4156)

[Lee, Taiyeong et al., 2013 Incremental Response Modeling Using SAS Enterprise Miner. SAS Global Forum 2013: Data Mining and Text Analytics.](https://support.sas.com/resources/papers/proceedings13/096-2013.pdf)

[Pylift: an uplift library that provides, primarily, (1) fast uplift modeling implementations and (2) evaluation tools. Accessed 7/5/2020.](https://pylift.readthedocs.io/en/latest/)

[Yi, Robert and Will Frost, 2018. Pylift: A Fast Python Package for Uplift Modeling. Accessed 7/5/2020.](https://tech.wayfair.com/data-science/2018/10/pylift-a-fast-python-package-for-uplift-modeling)

[Zhao, Zhenyu et al., 2020. Feature Selection Methods for Uplift Modeling.](https://arxiv.org/abs/2005.03447)
