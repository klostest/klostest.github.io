---
layout: post
title:  "Visualizing a balance history from transaction data using pandas and Plotly"
date:   2020-04-12
mathjax: true
permalink: /balance-history/
post_description: A practical post on data wrangling and visualization that could be applied in a professional setting, or to your personal finances.
post_image: "/assets/images/2020-04-12-balance-history/pexels-photo-209224.jpeg"
reading_time_minutes: 18
category: Technical
---

<img src="/assets/images/2020-04-12-balance-history/pexels-photo-209224.jpeg" alt="Accounting image" style="width: 350px; border: 10px solid white;" align="right"/>

# Introduction

Data science skills are good for exploring the world around us through a quantitative lens, and can also lead to a rewarding and stimulating career. But data science can also help you organize your personal business. For example, I was recently interested to see the balance history of my checking account. My transaction history is available for download from my bank, but the balance history is not.

However, I realized I could use the popular Python package [pandas](https://pandas.pydata.org/) to wrangle the transaction data and compute a balance history. Then I learned a visualization tool that I'd never used, but heard a lot of good things about, [Plotly](https://plotly.com/python/), to create an interactive graph of my balance history with all the information I wanted to see. This included a line plot of the daily balance, with hover text displaying the description and amount of individual transactions upon mouseover, as well as a plot of monthly average balance. In the end, I found the result quite helpful in giving me a quick and informative look at my financial history.

If you're interested to learn a bit more about wrangling and visualizing data in Python, I hope that you can gain some of those skills from reading this post. Perhaps you may even use the code for the same purpose I did. This notebook could be adapted to visualize a balance history using any dated transaction history. And if you happen to bank with a certain major national chain, you will find you can download your transaction history as a CSV file in exactly the same format as the synthetic data shown here, and use the code directly. Enjoy!

---

# Loading packages and printing versions
First things first.


```python
import sys
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
print('The Python version is {}\n'.format(sys.version))
print('The pandas version is {}\n'.format(pd.__version__))
print('The NumPy version is {}\n'.format(np.__version__))
print('The Plotly version is {}\n'.format(plotly.__version__))
```

    The Python version is 3.7.4 (default, Aug 13 2019, 15:17:50) 
    [Clang 4.0.1 (tags/RELEASE_401/final)]
    
    The pandas version is 1.0.1
    
    The NumPy version is 1.18.1
    
    The Plotly version is 4.5.4
    


# Loading and exploring the data

I've created synthetic transaction data for a person with a very simple financial profile: they get a bimonthly paycheck, pay rent and bills on the first business day of every month, and have a few other expenses at irregular times, including a large purchase that might be a car, for example. This is much simpler than what a real transaction record would probably look like, but it works for illustrating the approach here. If you're interested in the details of using pandas to create the synthetic data, please see the appendix below. 

Let's load the data and do some basic profiling.

__Note:__ I needed to use the `index_col=False` argument in `read_csv` to load the data I downloaded from my bank, as there was a missing column header for the first column.


```python
transactions_df = pd.read_csv('../data/synthetic_transaction_data.csv')
transactions_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 41 entries, 0 to 40
    Data columns (total 5 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   Unnamed: 0   41 non-null     object 
     1   Date         41 non-null     object 
     2   Description  41 non-null     object 
     3   Credit       24 non-null     float64
     4   Debit        17 non-null     float64
    dtypes: float64(2), object(3)
    memory usage: 1.7+ KB



```python
transactions_df.head()
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
      <th>Date</th>
      <th>Description</th>
      <th>Credit</th>
      <th>Debit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cleared</td>
      <td>2019-12-31</td>
      <td>Payroll</td>
      <td>2500.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cleared</td>
      <td>2019-12-15</td>
      <td>Payroll</td>
      <td>2500.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cleared</td>
      <td>2019-12-02</td>
      <td>Rent and bills</td>
      <td>NaN</td>
      <td>3500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cleared</td>
      <td>2019-11-30</td>
      <td>Payroll</td>
      <td>2500.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cleared</td>
      <td>2019-11-29</td>
      <td>Purchase</td>
      <td>NaN</td>
      <td>750.0</td>
    </tr>
  </tbody>
</table>
</div>



Looks like there are 41 rows in the data, and each row represents a transaction. The first column here looks like it consists entirely of the string `Cleared`:


```python
transactions_df['Unnamed: 0'].unique()
```




    array(['Cleared'], dtype=object)



Yes it does. This likely just means the transaction has cleared, or in other words been settled. The other columns include the date of the transaction, a description, and separate columns for whether the transaction was a credit to the account (i.e. money flowing in to the account) or a debit (money flowing out). Each transaction is either debit or credit but not both, with the unused column having a missing value in each row.

Convert the `Date` column to a `datetime` data type in pandas to enable pandas' powerful time series capabilities, then look at the range of dates.


```python
transactions_df['Date'] = pd.to_datetime(transactions_df['Date'])
print(transactions_df['Date'].min(), transactions_df['Date'].max())
```

    2019-01-01 00:00:00 2019-12-31 00:00:00


We can see the data span one year in time.

# Wrangling the data and calculating a balance history

Right now the transaction amounts are contained in separate columns. Ideally we'd have one column that includes all transaction amounts, with a sign to tell the difference between credits and debits. This would make it easier to add credits and subtract debits from the balance through time. Let's create such a column, by filling the missing values with zero, adding the credits, and subtracting the debits:


```python
transactions_df['Transaction'] = transactions_df['Credit'].fillna(0)\
- transactions_df['Debit'].fillna(0)
```


```python
transactions_df.head(5)
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
      <th>Date</th>
      <th>Description</th>
      <th>Credit</th>
      <th>Debit</th>
      <th>Transaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cleared</td>
      <td>2019-12-31</td>
      <td>Payroll</td>
      <td>2500.0</td>
      <td>NaN</td>
      <td>2500.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cleared</td>
      <td>2019-12-15</td>
      <td>Payroll</td>
      <td>2500.0</td>
      <td>NaN</td>
      <td>2500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cleared</td>
      <td>2019-12-02</td>
      <td>Rent and bills</td>
      <td>NaN</td>
      <td>3500.0</td>
      <td>-3500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cleared</td>
      <td>2019-11-30</td>
      <td>Payroll</td>
      <td>2500.0</td>
      <td>NaN</td>
      <td>2500.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cleared</td>
      <td>2019-11-29</td>
      <td>Purchase</td>
      <td>NaN</td>
      <td>750.0</td>
      <td>-750.0</td>
    </tr>
  </tbody>
</table>
</div>



Looks like this worked.

Now, one of the things I've seen people make with Plotly is an interactive plot with hover text, that appears over data points when the user mouses over them. I decided I wanted the amount and description of each transaction to appear as hover text on my plot, so I created a new column with this string.


```python
transactions_df['Description_amount'] = \
transactions_df['Transaction'].astype(str).str.cat(
    transactions_df['Description'], sep=': ')
```


```python
transactions_df.head()
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
      <th>Date</th>
      <th>Description</th>
      <th>Credit</th>
      <th>Debit</th>
      <th>Transaction</th>
      <th>Description_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cleared</td>
      <td>2019-12-31</td>
      <td>Payroll</td>
      <td>2500.0</td>
      <td>NaN</td>
      <td>2500.0</td>
      <td>2500.0: Payroll</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cleared</td>
      <td>2019-12-15</td>
      <td>Payroll</td>
      <td>2500.0</td>
      <td>NaN</td>
      <td>2500.0</td>
      <td>2500.0: Payroll</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cleared</td>
      <td>2019-12-02</td>
      <td>Rent and bills</td>
      <td>NaN</td>
      <td>3500.0</td>
      <td>-3500.0</td>
      <td>-3500.0: Rent and bills</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cleared</td>
      <td>2019-11-30</td>
      <td>Payroll</td>
      <td>2500.0</td>
      <td>NaN</td>
      <td>2500.0</td>
      <td>2500.0: Payroll</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cleared</td>
      <td>2019-11-29</td>
      <td>Purchase</td>
      <td>NaN</td>
      <td>750.0</td>
      <td>-750.0</td>
      <td>-750.0: Purchase</td>
    </tr>
  </tbody>
</table>
</div>



Looking at the earliest transactions in the record, which come last in the reverse chronological order of the data, we can see that some dates have more than one transaction.


```python
transactions_df.tail()
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
      <th>Date</th>
      <th>Description</th>
      <th>Credit</th>
      <th>Debit</th>
      <th>Transaction</th>
      <th>Description_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>36</th>
      <td>Cleared</td>
      <td>2019-02-01</td>
      <td>Rent and bills</td>
      <td>NaN</td>
      <td>3500.0</td>
      <td>-3500.0</td>
      <td>-3500.0: Rent and bills</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Cleared</td>
      <td>2019-01-31</td>
      <td>Payroll</td>
      <td>2200.0</td>
      <td>NaN</td>
      <td>2200.0</td>
      <td>2200.0: Payroll</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Cleared</td>
      <td>2019-01-15</td>
      <td>Purchase</td>
      <td>NaN</td>
      <td>1000.0</td>
      <td>-1000.0</td>
      <td>-1000.0: Purchase</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Cleared</td>
      <td>2019-01-15</td>
      <td>Payroll</td>
      <td>2200.0</td>
      <td>NaN</td>
      <td>2200.0</td>
      <td>2200.0: Payroll</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Cleared</td>
      <td>2019-01-01</td>
      <td>Rent and bills</td>
      <td>NaN</td>
      <td>3500.0</td>
      <td>-3500.0</td>
      <td>-3500.0: Rent and bills</td>
    </tr>
  </tbody>
</table>
</div>



In order to have a daily balance, we need to collapse the data down to a date level. In other words, we need to group by date. To aggregate the transactions, we'll take the sum on each date, giving the net change in balance on that day. Also, in order to have the hover text for each date, we need to combine the strings in some way. Concatenating strings with a newline separator is a readable way to do this, and it turns out `<br>` is the way to get a newline in Plotly text.


```python
transactions_date_group = transactions_df.groupby('Date').agg({
    'Description_amount': '<br>'.join,
    'Transaction':'sum'
})
```


```python
transactions_date_group.head()
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
      <th>Description_amount</th>
      <th>Transaction</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-01-01</th>
      <td>-3500.0: Rent and bills</td>
      <td>-3500.0</td>
    </tr>
    <tr>
      <th>2019-01-15</th>
      <td>-1000.0: Purchase&lt;br&gt;2200.0: Payroll</td>
      <td>1200.0</td>
    </tr>
    <tr>
      <th>2019-01-31</th>
      <td>2200.0: Payroll</td>
      <td>2200.0</td>
    </tr>
    <tr>
      <th>2019-02-01</th>
      <td>-3500.0: Rent and bills</td>
      <td>-3500.0</td>
    </tr>
    <tr>
      <th>2019-02-14</th>
      <td>-800.0: Purchase</td>
      <td>-800.0</td>
    </tr>
  </tbody>
</table>
</div>



The grouping by date resulted in a chronological ordering from earliest to latest, and we can see that multiple transactions on the same date have been combined. We are almost done with calculating a balance history. The last steps require knowing the balance at some point in time. This information should be readily available on the day the balance history was obtained. Let's say the hypothetical person here had $22,834 in the bank on 12/31/2019, after all transactions had cleared for that day.


```python
ending_balance = 22834
```

What about the starting balance? If we subtract all the credits and add all the debits in the record to the ending balance, we will have the balance before the first transaction in the record. In order to get this starting balance, we can subtract the sum of the `Transaction` column we created from the ending balance.


```python
starting_balance = ending_balance - transactions_date_group['Transaction'].sum()
starting_balance
```




    19984.0



To get a balance through time, we need the starting balance with all credits added, and debits subtracted, for each day in the record up to and including that day. This would result in a daily balance according to the clearing date of each transaction. In order to get a sum of all previous rows up to and including the current row, we can use pandas' `cumsum` method on the `Transaction` `Series`. Then we add this to the starting balance and we have the balance history.


```python
transactions_date_group['Running balance'] = starting_balance\
+ transactions_date_group['Transaction'].cumsum()
```


```python
transactions_date_group.head()
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
      <th>Description_amount</th>
      <th>Transaction</th>
      <th>Running balance</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-01-01</th>
      <td>-3500.0: Rent and bills</td>
      <td>-3500.0</td>
      <td>16484.0</td>
    </tr>
    <tr>
      <th>2019-01-15</th>
      <td>-1000.0: Purchase&lt;br&gt;2200.0: Payroll</td>
      <td>1200.0</td>
      <td>17684.0</td>
    </tr>
    <tr>
      <th>2019-01-31</th>
      <td>2200.0: Payroll</td>
      <td>2200.0</td>
      <td>19884.0</td>
    </tr>
    <tr>
      <th>2019-02-01</th>
      <td>-3500.0: Rent and bills</td>
      <td>-3500.0</td>
      <td>16384.0</td>
    </tr>
    <tr>
      <th>2019-02-14</th>
      <td>-800.0: Purchase</td>
      <td>-800.0</td>
      <td>15584.0</td>
    </tr>
  </tbody>
</table>
</div>



It looks like this has worked. We can check by confirming the running balance at the end of the first day in the record is the starting balance plus the sum of transactions for that day:


```python
first_date_index = transactions_date_group.index[0]
check_1 = starting_balance + transactions_date_group.loc[first_date_index,
                                                         'Transaction']
check_2 = transactions_date_group.loc[first_date_index, 'Running balance']

print(check_1)
print(check_2)

assert check_1 == check_2
```

    16484.0
    16484.0


Looks like our check passed! Now we have enough information to plot a daily balance with descriptive hover text. However, I also wanted to know what the monthly average balance was, where the average was taken over the end-of-day balance for all days in the month. For this we need a bit more data manipulation.

# Monthly average balance

Right now the frequency of our balance history is irregular. There are only records for days that had transactions. If we want an accurate monthly average, representing the average balance across all the days of the month, it would be helpful to interpolate our time series of balance history, filling in missing days with the most recent balance. A quick way to do this in pandas is to take our balance history that we already calculated and create a new `DataFrame` with a `DatetimeIndex` that we specify to have all the days of the year. Then we can easily fill in missing values to have the daily balance.

First let's create the daily index.


```python
daily_index = pd.date_range(transactions_date_group.index.min(),
                            transactions_date_group.index.max())
```

Now let's create the new `DataFrame` and `reindex` it using our daily index. The days with no transaction data will be filled with `np.nan`.


```python
df_daily = pd.DataFrame(transactions_date_group['Running balance'])
df_daily = df_daily.reindex(daily_index, fill_value=np.nan)
```


```python
df_daily.head(5)
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
      <th>Running balance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-01-01</th>
      <td>16484.0</td>
    </tr>
    <tr>
      <th>2019-01-02</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-01-03</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-01-04</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-01-05</th>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Looks like we've got a row for each day! Now, we need to fill missing values, with the most recent balance. Pandas has an option to "forward fill" missing values, which works well with our chronologically ordered `DataFrame`.


```python
df_daily = df_daily.fillna(method='ffill')
```


```python
df_daily.head(5)
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
      <th>Running balance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-01-01</th>
      <td>16484.0</td>
    </tr>
    <tr>
      <th>2019-01-02</th>
      <td>16484.0</td>
    </tr>
    <tr>
      <th>2019-01-03</th>
      <td>16484.0</td>
    </tr>
    <tr>
      <th>2019-01-04</th>
      <td>16484.0</td>
    </tr>
    <tr>
      <th>2019-01-05</th>
      <td>16484.0</td>
    </tr>
  </tbody>
</table>
</div>



Finally, because we have a `DatetimeIndex`, calcuating the monthly average balance is a snap with pandas resampling capabilities.


```python
df_monthly = \
pd.DataFrame(df_daily['Running balance'].resample(rule='1M').mean())
```


```python
df_monthly.head()
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
      <th>Running balance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-01-31</th>
      <td>17213.032258</td>
    </tr>
    <tr>
      <th>2019-02-28</th>
      <td>17134.000000</td>
    </tr>
    <tr>
      <th>2019-03-31</th>
      <td>17761.419355</td>
    </tr>
    <tr>
      <th>2019-04-30</th>
      <td>18800.666667</td>
    </tr>
    <tr>
      <th>2019-05-31</th>
      <td>20335.612903</td>
    </tr>
  </tbody>
</table>
</div>



This has given us the average monthly balance, associated with the last day of the month.

# Creating the plot
We now have all the information we need to create a descriptive plot of daily balance and monthly average balance. In order to do this, I found that Plotly's Graph Objects API provided the flexibility I needed to customize the plot to my liking, although Plotly has another more lightweight interface called Plotly Express.

Unpacking the code below, we first create a figure, similar to plotting with Matplotlib, another popular visualization library in Python. Then we add two "traces" to the figure, each of which is a `Scatter` object. These are the plots of daily and monthly average balance.

The `DatetimeIndex` from each `DataFrame` we created is used for the x-coordinates in each plot, and the y-coordinates are the quantities we want to visualize over time, with some rounding for visual presentation purposes. Both plots will have lines and markers (`mode='lines+markers'`), and are named so they can be represented in the legend.

Aside from these specifications, the daily balance plot has additional options including the custom hover text we created (`text`), and a `hoverinfo` argument stating that we want the x- and y-coordinates displayed upon mouseover, along with the custom text. Finally this plot is given an `hv` line shape, which means that in order to trace the line from point to point through time, first a step is taken in the horizontal direction, then the vertical direction. This is what we want, because the balance remains the same after each day with a transaction, moving horizontally through time on the graph, and then moves vertically to the new balance on the day of the next transaction. The effect of this is similar to filling missing values forward in time, like we did with the daily-interpolated balances above.

The last touch before `show`ing the figure is to format the hover text, so that it doesn't show too many decimal places. And then "Voila!", we have our visualization of balance history!

```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=transactions_date_group.index,
                         y=transactions_date_group['Running balance'].round(0),
                         mode='lines+markers',
                         name='Daily balance',
                         text=transactions_date_group['Description_amount'],
                         hoverinfo='x+y+text',
                         line_shape='hv'))

fig.add_trace(go.Scatter(x=df_monthly.index,
                         y=df_monthly['Running balance'].round(0),
                         mode='lines+markers',
                         name='Monthly mean balance'))

fig.update_yaxes(hoverformat="$d")

fig.show()
```

{% include balance-history-plot.html %}

Notice how you can interact with the plot, including mousing over to see details, as well as dragging to select a portion to zoom in on. Double-click to zoom back out.

As you can see, there are many ways to customize visualizations with Plotly and I've likely only scratched the surface here. For me, the most important part was the hover text that I was able to customize to show all the information I wanted. But I also realized it's nice to be able to zoom in to particular regions of the plot to get a more detailed look.

If you were looking to learn more about data wrangling with pandas and visualization with Plotly, or even do the specific task performed here, I hope you found this blog post helpful. For more data science and machine learning resources, check out my other posts as well as my book [Data Science Projects with Python: A case study approach to successful data science projects using Python, pandas, and scikit-learn](https://www.amazon.com/Data-Science-Projects-Python-scikit-learn-dp-1838551026/dp/1838551026/).

# Appendix: Creating the synthetic data
Start with a bimonthly paycheck (`SM` = semi-month end frequency (15th and end of month)). This person got a raise starting in April. Good for them!


```python
paycheck_times = pd.Series(pd.date_range(start=pd.Timestamp(2019,1,1),
                                         end=pd.Timestamp(2019,12,31),
                                         freq='SM'))
paycheck_amounts = np.empty(paycheck_times.shape)
paycheck_amounts[:6] = 2200
paycheck_amounts[6:] = 2500
paycheck_df = pd.DataFrame({'Unnamed: 0':'Cleared',
                            'Date':paycheck_times,
                            'Description':'Payroll',
                            'Credit':paycheck_amounts})
paycheck_df.head(8)
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
      <th>Date</th>
      <th>Description</th>
      <th>Credit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cleared</td>
      <td>2019-01-15</td>
      <td>Payroll</td>
      <td>2200.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cleared</td>
      <td>2019-01-31</td>
      <td>Payroll</td>
      <td>2200.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cleared</td>
      <td>2019-02-15</td>
      <td>Payroll</td>
      <td>2200.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cleared</td>
      <td>2019-02-28</td>
      <td>Payroll</td>
      <td>2200.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cleared</td>
      <td>2019-03-15</td>
      <td>Payroll</td>
      <td>2200.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Cleared</td>
      <td>2019-03-31</td>
      <td>Payroll</td>
      <td>2200.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Cleared</td>
      <td>2019-04-15</td>
      <td>Payroll</td>
      <td>2500.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Cleared</td>
      <td>2019-04-30</td>
      <td>Payroll</td>
      <td>2500.0</td>
    </tr>
  </tbody>
</table>
</div>



Include some rent. For simplicity's sake, I'll assume all this person's monthly expenses come on the first business day of the month (`BMS` or business month start), although a realistic transaction record would likely be more complicated.


```python
rent_times = pd.Series(pd.date_range(start=pd.Timestamp(2019,1,1),
                                     end=pd.Timestamp(2019,12,31),
                                     freq='BMS'))
rent_amounts = np.ones(rent_times.shape) * 3500
rent_df = pd.DataFrame({'Unnamed: 0':'Cleared',
                        'Date':rent_times,
                        'Description':'Rent and bills',
                        'Debit':rent_amounts})
rent_df.head()
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
      <th>Date</th>
      <th>Description</th>
      <th>Debit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cleared</td>
      <td>2019-01-01</td>
      <td>Rent and bills</td>
      <td>3500.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cleared</td>
      <td>2019-02-01</td>
      <td>Rent and bills</td>
      <td>3500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cleared</td>
      <td>2019-03-01</td>
      <td>Rent and bills</td>
      <td>3500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cleared</td>
      <td>2019-04-01</td>
      <td>Rent and bills</td>
      <td>3500.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cleared</td>
      <td>2019-05-01</td>
      <td>Rent and bills</td>
      <td>3500.0</td>
    </tr>
  </tbody>
</table>
</div>



Finally, some purchases.


```python
purchase_times = pd.Series([
    pd.Timestamp(2019,1,15),
    pd.Timestamp(2019,2,14),
    pd.Timestamp(2019,7,1),
    pd.Timestamp(2019,9,5),
    pd.Timestamp(2019,11,29)
])
purchase_amounts = np.array([1000, 800, 10500, 300, 750])
purchase_df = pd.DataFrame({'Unnamed: 0':'Cleared',
                            'Date':purchase_times,
                            'Description':'Purchase',
                            'Debit':purchase_amounts})
purchase_df.head()
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
      <th>Date</th>
      <th>Description</th>
      <th>Debit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cleared</td>
      <td>2019-01-15</td>
      <td>Purchase</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cleared</td>
      <td>2019-02-14</td>
      <td>Purchase</td>
      <td>800</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cleared</td>
      <td>2019-07-01</td>
      <td>Purchase</td>
      <td>10500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cleared</td>
      <td>2019-09-05</td>
      <td>Purchase</td>
      <td>300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cleared</td>
      <td>2019-11-29</td>
      <td>Purchase</td>
      <td>750</td>
    </tr>
  </tbody>
</table>
</div>



Merge all transactions into one `DataFrame`.


```python
transactions_df = pd.concat([paycheck_df, rent_df, purchase_df])
transactions_df = transactions_df.sort_values(by='Date',
                                              ascending=False,
                                              ignore_index=True)
transactions_df.head()
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
      <th>Date</th>
      <th>Description</th>
      <th>Credit</th>
      <th>Debit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cleared</td>
      <td>2019-12-31</td>
      <td>Payroll</td>
      <td>2500.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cleared</td>
      <td>2019-12-15</td>
      <td>Payroll</td>
      <td>2500.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cleared</td>
      <td>2019-12-02</td>
      <td>Rent and bills</td>
      <td>NaN</td>
      <td>3500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cleared</td>
      <td>2019-11-30</td>
      <td>Payroll</td>
      <td>2500.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cleared</td>
      <td>2019-11-29</td>
      <td>Purchase</td>
      <td>NaN</td>
      <td>750.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
transactions_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 41 entries, 0 to 40
    Data columns (total 5 columns):
     #   Column       Non-Null Count  Dtype         
    ---  ------       --------------  -----         
     0   Unnamed: 0   41 non-null     object        
     1   Date         41 non-null     datetime64[ns]
     2   Description  41 non-null     object        
     3   Credit       24 non-null     float64       
     4   Debit        17 non-null     float64       
    dtypes: datetime64[ns](1), float64(2), object(2)
    memory usage: 1.7+ KB



```python
transactions_df.to_csv('../data/synthetic_transaction_data.csv', index=False)
```
