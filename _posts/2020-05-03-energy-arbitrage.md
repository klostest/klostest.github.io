---
layout: post
title:  "Designing an energy arbitrage strategy with linear programming"
date:   2020-05-03
permalink: /energy-arbitrage/
post_description: How to use a classic mathematical optimization technique to guide the operation of a grid-connected battery and maximize profit.
post_image: "/assets/images/2020-05-03-energy-arbitrage/cable-clouds-conductor-current-189524.jpg"
reading_time_minutes: 30
category: Technical
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<img src="/assets/images/2020-05-03-energy-arbitrage/cable-clouds-conductor-current-189524.jpg" alt="Power lines, Photo by Pok Rie from Pexels, https://www.pexels.com/photo/cable-clouds-conductor-current-189524/" style="width: 350px; border: 10px solid white;" align="right"/>

The price of energy changes hourly, which opens up the possibility of temporal arbitrage: buying energy at a low price, storing it, and selling it later at a higher price. To successfully execute any temporal arbitrage strategy, some amount of confidence in future prices is required, to be able to expect to make a profit. In the case of energy arbitrage, the constraints of the energy storage system must also be considered. For example, batteries have limited capacity, limited rate of charging, and are not 100% efficient in that not all of the energy used to charge a battery will be available later for discharge.

The goal of this post is to illustrate how the mathematical optimization technique of linear programming can be used to design an operating strategy for a grid-connected battery with a given set of operational constraints, under the assumption of known future prices. All of this will be done using real-world energy price data and open source tools in Python. The Jupyter notebook is available [here](https://github.com/klostest/energy_arbitrage_work_sample/blob/master/Energy%20arbitrage.ipynb).

# Problem and approach

The task is to formulate an operation strategy for a grid connected battery system, to perform energy arbitrage on the New York Independent System Operator (NYISO) day-ahead market. What is the NYISO? According to their [website](https://www.nyiso.com/what-we-do),
>The NYISO is the New York Independent System Operator — the organization responsible for managing New York’s electric grid and its competitive wholesale electric marketplace.

The NYISO makes the next day's hourly energy prices available at 11am each day (NYISO 2019, references available at bottom of page). The battery system we design here will schedule the next 24 hours of battery operation (noon of the current day through the 11am hour of the next day) using this information. We will assume that after the daily price announcement from the NYISO, the next 36 hours of price information are available: the prices for the remaining 12 hours of the current day, as well as 24 hours worth of prices for the next day. Therefore the optimization time horizon will be 36 hours, to take advantage of all available future price information.

Since the operational schedule will be repeated each day, the last 12 hours of the 36 hour strategy will always be ignored. This is because new price information will become available at 11am the following day, which we can take advantage of. However these "extra 12 hours" of data are not wasted; I determined in initial experiments that having a 36 hour time horizon creates a more profitable arbitrage strategy than shorter horizons. This makes intuitive sense, because the more future information we can incorporate into the arbitrage strategy, the more profitable it should be. You can experiment with different horizons using the code below, although the horizon is assumed to be at least 24 hours here. The plan can be visualized as follows:

![png](/assets/images/2020-05-03-energy-arbitrage/Slide2_crop.png)

The battery is said to be a price taker, meaning its activities do not affect the price of energy. The price paid for power to charge the battery, and revenue from discharging, is the location based marginal price (LBMP), which takes in to account the system marginal price, congestion component, and marginal loss component (PJM Interconnection LLC). The goal is to maximize profit, given the day-ahead prices and the battery system's parameters.

In this scenario, where future prices are known and the battery system is a price taker, the problem of designing an operational strategy can be solved by linear programming (Salles et al. 2017, Sioshansi et al. 2009, Wang and Zhang 2018). In brief summary, [linear programming](https://en.wikipedia.org/wiki/Linear_programming) is a well-known technique for either maximizing or minimizing some objective. In this case, we want to maximize profit. As long as the mathematical function describing the objective, known as the __objective function__, as well as the __constraints__ of the system, can all be described as linear combinations of the __decision variables__, which define the operational strategy, linear programming can be used to optimize the system.

## Setting up the linear programming problem in PuLP

Loosely following the notation of Sioshansi et al. (2009), here we'll lay out the decision variables and add the contraints to a linear programming model in PuLP. Starting with the class definition in the cell after next, the markdown code snippets in this section should all be put together to define a class, which describes our battery system (see the [notebook](https://github.com/klostest/energy_arbitrage_work_sample/blob/master/Energy%20arbitrage.ipynb) for confirmation). This model of the system will be useful to simulate battery operation, stepping through time at a daily increment.

Before proceeding further let's import the packages needed for this exercise.


```python
#Load packages
import pulp
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
%matplotlib inline 
```

Each day, a single optimization problem needs to be solved between 11am and 12pm, which will provide sufficient information to guide the battery's operation for the next 36 hours. To guide the battery's operation, we need to decide what the flow of energy will be for each discrete time step in the time horizon. Energy flows, also known as electrical power, can either be into or out of the battery. So we'll create two decision variables, $$c_t$$ and $$d_t$$, as the charging and discharging power flows, respectively (kW), at time $$t$$, which will be an hourly time step. The rates of charge or discharge are continuous variables and are bounded to be within the operating limits of the battery, for all time steps:

$$ 0 \le c_t, d_t \le \kappa, \forall t $$

where $$\kappa$$ (kW) is the maximum charge and discharge power capacity, which we'll assume are equal here. 

We need to specify flow variables for each time step in the time horizon of optimization. PuLP provides a handy method `dicts` of the `LpVariable` class, which we can use to create charge and discharge flows for all time steps at once. We'll start defining a class that will hold all the information governing the operation of the battery. The input `time_horizon` is an integer specifiying the number of hours in the optimization horizon, assumed here to be at least 24, and the other inputs are as described above:

```python
class Battery():
    
    def __init__(self,
                 time_horizon,
                 max_discharge_power_capacity,
                 max_charge_power_capacity):
        #Set up decision variables for optimization.
        #These are the hourly charge and discharge flows for
        #the optimization horizon, with their limitations.
        self.time_horizon = time_horizon
    
        self.charge = \
        pulp.LpVariable.dicts(
            "charging_power",
            ('c_t_' + str(i) for i in range(0,time_horizon)),
            lowBound=0, upBound=max_charge_power_capacity,
            cat='Continuous')

        self.discharge = \
        pulp.LpVariable.dicts(
            "discharging_power",
            ('d_t_' + str(i) for i in range(0,time_horizon)),
            lowBound=0, upBound=max_discharge_power_capacity,
            cat='Continuous')
```

After setting up all the decision variables, it's time to define the optimization problem that PuLP will solve for us. Our goal is to maximize profit $$P$$ within the optimization time horizon, which can be defined as follows using charge flows and the price of energy:

$$ P \left( d_t, c_t, p_t \right) = \frac{ \sum_{t=12\text{pm of current day}}^{11\text{pm of next day}} p_t \cdot (d_t - c_t) }{1000}$$

where $$p_t$$ is the LBMP (\$/MWh) at time $$t$$. In terms of units, because $$t$$ is an hourly time step, it effectively cancels the per hour part of the units of the LBMP. The objective function is divided by 1000 to account for the discrepancy in units (MW/kW), so ultimately the objective of profit will be in units of dollars. We'll make this unit correction after running the simulation, so it won't be reflected in the code here.

The objective function is specified by adding the decision variables to the model object in PuLP, multiplying each by the appropriate cost or revenue that would come from charging or discharging that amount of energy. This multiplication of decision variables by prices is accomplished using an `LpAffineExpression`. The `prices` are determined by the LBMP, which we'll select for the relevant time period when we run the simulation. For example, with a 36 hour time horizon, this would be noon the current day, through the 11pm hour the next day.

```python
    def set_objective(self, prices):
        #Create a model and objective function.
        #This uses price data, which must have one price
        #for each point in the time horizon.
        try:
            assert len(prices) == self.time_horizon
        except:
            print('Error: need one price for each hour in time horizon')
        
        #Instantiate linear programming model to maximize the objective
        self.model = pulp.LpProblem("Energy arbitrage", pulp.LpMaximize)
    
        #Objective is profit
        #This formula gives the daily profit from charging/discharging
        #activities. Charging is a cost, discharging is a revenue
        self.model += \
        pulp.LpAffineExpression(
            [(self.charge['c_t_' + str(i)],
              -1*prices[i]) for i in range(0,self.time_horizon)]) +\
        pulp.LpAffineExpression(
            [(self.discharge['d_t_' + str(i)],
              prices[i]) for i in range(0,self.time_horizon)])
```

Having defined the model and the objective, now we need to add the battery's operational constraints. The battery has limited capacity and so the optimization is subject to a battery storage constraint:

$$ 0 <= \sum_{t_f = 1\text{pm of current day}}^{11\text{pm of next day}} \sum_{t=12\text{pm of current day}}^{t_f} s_{i} + \eta \cdot
c_t - d_t <= \text{discharge energy capacity (kWh)}$$

where $$s_i$$ is the state of energy (kWh) of the battery at the start of the 36-hour optimization period and $$\eta$$ is the round-trip efficiency of the battery. This constraint requires the battery's state of energy (sum of initial state and hourly power flows) to be between zero, assuming the battery has complete depth of discharge capabilities, and the battery's discharge energy capacity, for each hour of the optimization horizon. In this constraint, the power flows (kW) are understood to be converted to units of energy (kWh) through multiplication by the one hour time step.

In PuLP, constraints can be added to a model just like the objective function: using addition syntax. We can express these constraints by adding up the discharge flows using `lpSum`, and in the case of charge flows which need to be multiplied by efficiency, again an `LpAffineExpression`.

```python
    def add_storage_constraints(self,
                                efficiency,
                                min_capacity,
                                discharge_energy_capacity,
                                initial_level):
        #Storage level constraint 1
        #This says the battery cannot have less than zero energy, at
        #any hour in the horizon
        #Note this is a place where round-trip efficiency is factored in.
        #The energy available for discharge is the round-trip efficiency
        #times the energy that was charged.       
        for hour_of_sim in range(1,self.time_horizon+1):     
            self.model += \
            initial_level \
            + pulp.LpAffineExpression(
                [(self.charge['c_t_' + str(i)], efficiency)
                 for i in range(0,hour_of_sim)]) \
            - pulp.lpSum(
                self.discharge[index]
                for index in('d_t_' + str(i)
                             for i in range(0,hour_of_sim)))\
            >= min_capacity
            
        #Storage level constraint 2
        #Similar to 1
        #This says the battery cannot have more than the
        #discharge energy capacity
        for hour_of_sim in range(1,self.time_horizon+1):
            self.model += \
            initial_level \
            + pulp.LpAffineExpression(
                [(self.charge['c_t_' + str(i)], efficiency)
                 for i in range(0,hour_of_sim)]) \
            - pulp.lpSum(
                self.discharge[index]
                for index in ('d_t_' + str(i)
                              for i in range(0,hour_of_sim)))\
            <= discharge_energy_capacity
```

Maximum daily discharged throughput $$\tau$$ (kWh) is also constrained, which limits the amount of energy that can flow through the battery in a given day. We set this up so that the first day of the time horizon is subject to a 24 hour constraint, and whatever portion beyond that is subject to a fractional constraint. For example, in our 36 hour horizon, the constraints would be:

$$ \sum_{t=12\text{pm of current day}}^{11\text{am of next day}} d_t <= \tau$$

$$ \sum_{t=12\text{pm of next day}}^{11\text{pm of next day}} d_t <= 0.5 \cdot \tau$$

```python
    def add_throughput_constraints(self,
                                   max_daily_discharged_throughput):
        #Maximum discharge throughput constraint
        #The sum of all discharge flow within a day cannot exceed this
        #Include portion of the next day according to time horizon
        #Assumes the time horizon is at least 24 hours
        
        self.model += \
        pulp.lpSum(
            self.discharge[index] for index in (
                'd_t_' + str(i) for i in range(0,24))) \
        <= max_daily_discharged_throughput
        
        self.model += \
        pulp.lpSum(
            self.discharge[index] for index in (
                'd_t_' + str(i) for i in range(25,self.time_horizon))) \
        <= max_daily_discharged_throughput \
        *float(self.time_horizon-24)/24
```

Now that we've set up the model with an objective function and all constraints, we include methods to solve the problem, and report back the results, which are the optimal charge and discharge flows for each hour in the time horizon. As long as the problem we set up is feasible in terms of the constraints we've indicated, linear programming should work in that it will find an optimal solution, that maximizes profit. However if not, we will return a message indicating this.

```python
    def solve_model(self):
        #Solve the optimization problem
        self.model.solve()
        
        #Show a warning if an optimal solution was not found
        if pulp.LpStatus[self.model.status] != 'Optimal':
            print('Warning: ' + pulp.LpStatus[self.model.status])
            
    def collect_output(self):  
        #Collect hourly charging and discharging rates within the
        #time horizon
        hourly_charges =\
            np.array(
                [self.charge[index].varValue for
                 index in ('c_t_' + str(i) for i in range(0,24))])
        hourly_discharges =\
            np.array(
                [self.discharge[index].varValue for
                 index in ('d_t_' + str(i) for i in range(0,24))])

        return hourly_charges, hourly_discharges
```

This completes the `Battery` class. We are now ready to ingest the data and proceed to simulating battery operation.

# Import price data
We obtain one year's worth of LBMPs, so we can simulate the battery's operation over this period of time. The data is available as LBMPs for several zones, at an hourly time step. Here we load the CSV files (one per day) and concatenate them on to a `DataFrame`. You can obtain these data from the [git repo](https://github.com/klostest/energy_arbitrage_work_sample) accompanying this blog post. The data were downloaded from [here](http://mis.nyiso.com/public/) on May 2, 2020, as zipped directories of CSV files (Pricing Data, Day-Ahead Market (DAM) LBMP, Zonal P-2A).


```python
#Directory of data
data_dir = './data_2019_2020_from_web/'
```


```python
dir_list = os.listdir(data_dir)
dir_list.sort()
dir_list
```




    ['.DS_Store',
     '20190501damlbmp_zone_csv',
     '20190601damlbmp_zone_csv',
     '20190701damlbmp_zone_csv',
     '20190801damlbmp_zone_csv',
     '20190901damlbmp_zone_csv',
     '20191001damlbmp_zone_csv',
     '20191101damlbmp_zone_csv',
     '20191201damlbmp_zone_csv',
     '20200101damlbmp_zone_csv',
     '20200201damlbmp_zone_csv',
     '20200301damlbmp_zone_csv',
     '20200401damlbmp_zone_csv']




```python
#Remove invisible files (i.e. .DS_Store used by Mac OS)
for this_item in dir_list:
    if this_item[0] == '.':
        dir_list.remove(this_item)
```

Loop through all the subdirectories, loading all the CSV files.


```python
tic = time.time()
#count loaded files
file_counter = 0

#For each subdirectory in the parent directory
for this_sub_dir in dir_list:
    #List the files
    this_sub_dir_list = os.listdir(data_dir + '/' + this_sub_dir)
    #Sort the list
    this_sub_dir_list.sort()
    #Delete invisible files (that start with '.')
    for this_item in this_sub_dir_list:
        if this_item[0] == '.':
            this_sub_dir_list.remove(this_item)
    #For each file in the subdirectory
    for this_file in this_sub_dir_list:
        #Load the contents into a DataFrame
        this_df = pd.read_csv(data_dir + '/' + this_sub_dir + '/' + this_file)
        #Concatenate with existing data if past first file
        if file_counter == 0:
            all_data = this_df.copy()
        else:
            all_data = pd.concat([all_data, this_df])
        
        file_counter += 1
toc = time.time()
print(str(toc-tic) + ' seconds run time')
```

    2.1731250286102295 seconds run time


Examine the data


```python
all_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 131760 entries, 0 to 359
    Data columns (total 6 columns):
     #   Column                             Non-Null Count   Dtype  
    ---  ------                             --------------   -----  
     0   Time Stamp                         131760 non-null  object 
     1   Name                               131760 non-null  object 
     2   PTID                               131760 non-null  int64  
     3   LBMP ($/MWHr)                      131760 non-null  float64
     4   Marginal Cost Losses ($/MWHr)      131760 non-null  float64
     5   Marginal Cost Congestion ($/MWHr)  131760 non-null  float64
    dtypes: float64(3), int64(1), object(2)
    memory usage: 7.0+ MB



```python
all_data.head()
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
      <th>Time Stamp</th>
      <th>Name</th>
      <th>PTID</th>
      <th>LBMP ($/MWHr)</th>
      <th>Marginal Cost Losses ($/MWHr)</th>
      <th>Marginal Cost Congestion ($/MWHr)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>05/01/2019 00:00</td>
      <td>CAPITL</td>
      <td>61757</td>
      <td>20.43</td>
      <td>0.93</td>
      <td>-4.04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>05/01/2019 00:00</td>
      <td>CENTRL</td>
      <td>61754</td>
      <td>16.17</td>
      <td>0.15</td>
      <td>-0.55</td>
    </tr>
    <tr>
      <th>2</th>
      <td>05/01/2019 00:00</td>
      <td>DUNWOD</td>
      <td>61760</td>
      <td>20.13</td>
      <td>1.50</td>
      <td>-3.17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>05/01/2019 00:00</td>
      <td>GENESE</td>
      <td>61753</td>
      <td>15.62</td>
      <td>-0.26</td>
      <td>-0.43</td>
    </tr>
    <tr>
      <th>4</th>
      <td>05/01/2019 00:00</td>
      <td>H Q</td>
      <td>61844</td>
      <td>15.09</td>
      <td>-0.37</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



Sanity checks on data. Considering we have 12 months of data including a February 29th from a leap year, were 366 days of data loaded?


```python
assert file_counter == 366
```

How many zones are there, and what are they?


```python
unique_names = all_data['Name'].unique()
print(len(unique_names))
unique_names
```

    15

    array(['CAPITL', 'CENTRL', 'DUNWOD', 'GENESE', 'H Q', 'HUD VL', 'LONGIL',
           'MHK VL', 'MILLWD', 'N.Y.C.', 'NORTH', 'NPX', 'O H', 'PJM', 'WEST'],
          dtype=object)



How may rows are there?


```python
all_data.shape
```




    (131760, 6)



Check that number of rows = number of zones times 24 hours a day times 366 days a year:


```python
assert 15*24*366 == all_data.shape[0]
```

For this example we'll concern ourselves only with New York City. Select out the data of interest (specific zone):


```python
zone_of_interest = 'N.Y.C.'
all_data = all_data.loc[all_data['Name'].isin([zone_of_interest]),:]
```


```python
all_data.shape
```




    (8784, 6)



The `DataFrame` will be easier to use with a `DatetimeIndex`.
Reset the index to the time stamp:


```python
all_data = all_data.set_index(['Time Stamp'])
```

Cast as `datetime`:


```python
all_data.index = pd.to_datetime(all_data.index, format='%m/%d/%Y %H:%M')
```

Let's examine how the data look around the daylight savings transition:


```python
start_time = pd.Timestamp(year=2019, month=11, day=2, hour=23)
end_time = pd.Timestamp(year=2019, month=11, day=3, hour=3)
```


```python
time_test_1 = all_data[start_time:end_time]
time_test_1
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
      <th>Name</th>
      <th>PTID</th>
      <th>LBMP ($/MWHr)</th>
      <th>Marginal Cost Losses ($/MWHr)</th>
      <th>Marginal Cost Congestion ($/MWHr)</th>
    </tr>
    <tr>
      <th>Time Stamp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-11-02 23:00:00</th>
      <td>N.Y.C.</td>
      <td>61761</td>
      <td>18.17</td>
      <td>1.15</td>
      <td>-5.74</td>
    </tr>
    <tr>
      <th>2019-11-03 00:00:00</th>
      <td>N.Y.C.</td>
      <td>61761</td>
      <td>19.03</td>
      <td>1.09</td>
      <td>-6.85</td>
    </tr>
    <tr>
      <th>2019-11-03 01:00:00</th>
      <td>N.Y.C.</td>
      <td>61761</td>
      <td>17.44</td>
      <td>0.99</td>
      <td>-6.21</td>
    </tr>
    <tr>
      <th>2019-11-03 01:00:00</th>
      <td>N.Y.C.</td>
      <td>61761</td>
      <td>17.35</td>
      <td>1.02</td>
      <td>-5.72</td>
    </tr>
    <tr>
      <th>2019-11-03 02:00:00</th>
      <td>N.Y.C.</td>
      <td>61761</td>
      <td>16.64</td>
      <td>0.87</td>
      <td>-6.45</td>
    </tr>
    <tr>
      <th>2019-11-03 03:00:00</th>
      <td>N.Y.C.</td>
      <td>61761</td>
      <td>16.23</td>
      <td>0.93</td>
      <td>-5.70</td>
    </tr>
  </tbody>
</table>
</div>



We can see that there are two entries for 1am, where the second one was a result of the clocks being "turned back". In order to do arithmetic with this `DatetimeIndex`, we need to make it timezone-aware. Pandas makes this easy and handles the duplicate 1am row appropriately:


```python
all_data.index = \
all_data.index.tz_localize('America/New_York', ambiguous='infer')
```


```python
time_test_2 = all_data[start_time:end_time]
time_test_2
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
      <th>Name</th>
      <th>PTID</th>
      <th>LBMP ($/MWHr)</th>
      <th>Marginal Cost Losses ($/MWHr)</th>
      <th>Marginal Cost Congestion ($/MWHr)</th>
    </tr>
    <tr>
      <th>Time Stamp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-11-02 23:00:00-04:00</th>
      <td>N.Y.C.</td>
      <td>61761</td>
      <td>18.17</td>
      <td>1.15</td>
      <td>-5.74</td>
    </tr>
    <tr>
      <th>2019-11-03 00:00:00-04:00</th>
      <td>N.Y.C.</td>
      <td>61761</td>
      <td>19.03</td>
      <td>1.09</td>
      <td>-6.85</td>
    </tr>
    <tr>
      <th>2019-11-03 01:00:00-04:00</th>
      <td>N.Y.C.</td>
      <td>61761</td>
      <td>17.44</td>
      <td>0.99</td>
      <td>-6.21</td>
    </tr>
    <tr>
      <th>2019-11-03 01:00:00-05:00</th>
      <td>N.Y.C.</td>
      <td>61761</td>
      <td>17.35</td>
      <td>1.02</td>
      <td>-5.72</td>
    </tr>
    <tr>
      <th>2019-11-03 02:00:00-05:00</th>
      <td>N.Y.C.</td>
      <td>61761</td>
      <td>16.64</td>
      <td>0.87</td>
      <td>-6.45</td>
    </tr>
    <tr>
      <th>2019-11-03 03:00:00-05:00</th>
      <td>N.Y.C.</td>
      <td>61761</td>
      <td>16.23</td>
      <td>0.93</td>
      <td>-5.70</td>
    </tr>
  </tbody>
</table>
</div>



Now we can see that the offset from UTC is indicated. Let's double check the beginning and end of our data before proceeding.


```python
all_data.head()
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
      <th>Name</th>
      <th>PTID</th>
      <th>LBMP ($/MWHr)</th>
      <th>Marginal Cost Losses ($/MWHr)</th>
      <th>Marginal Cost Congestion ($/MWHr)</th>
    </tr>
    <tr>
      <th>Time Stamp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-05-01 00:00:00-04:00</th>
      <td>N.Y.C.</td>
      <td>61761</td>
      <td>20.32</td>
      <td>1.69</td>
      <td>-3.18</td>
    </tr>
    <tr>
      <th>2019-05-01 01:00:00-04:00</th>
      <td>N.Y.C.</td>
      <td>61761</td>
      <td>17.09</td>
      <td>1.62</td>
      <td>-0.06</td>
    </tr>
    <tr>
      <th>2019-05-01 02:00:00-04:00</th>
      <td>N.Y.C.</td>
      <td>61761</td>
      <td>14.89</td>
      <td>1.37</td>
      <td>-0.05</td>
    </tr>
    <tr>
      <th>2019-05-01 03:00:00-04:00</th>
      <td>N.Y.C.</td>
      <td>61761</td>
      <td>14.49</td>
      <td>1.29</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2019-05-01 04:00:00-04:00</th>
      <td>N.Y.C.</td>
      <td>61761</td>
      <td>14.49</td>
      <td>1.27</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_data.tail()
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
      <th>Name</th>
      <th>PTID</th>
      <th>LBMP ($/MWHr)</th>
      <th>Marginal Cost Losses ($/MWHr)</th>
      <th>Marginal Cost Congestion ($/MWHr)</th>
    </tr>
    <tr>
      <th>Time Stamp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-04-30 19:00:00-04:00</th>
      <td>N.Y.C.</td>
      <td>61761</td>
      <td>18.92</td>
      <td>1.49</td>
      <td>-4.15</td>
    </tr>
    <tr>
      <th>2020-04-30 20:00:00-04:00</th>
      <td>N.Y.C.</td>
      <td>61761</td>
      <td>20.91</td>
      <td>1.52</td>
      <td>-5.60</td>
    </tr>
    <tr>
      <th>2020-04-30 21:00:00-04:00</th>
      <td>N.Y.C.</td>
      <td>61761</td>
      <td>18.08</td>
      <td>1.48</td>
      <td>-3.36</td>
    </tr>
    <tr>
      <th>2020-04-30 22:00:00-04:00</th>
      <td>N.Y.C.</td>
      <td>61761</td>
      <td>16.36</td>
      <td>1.45</td>
      <td>-1.75</td>
    </tr>
    <tr>
      <th>2020-04-30 23:00:00-04:00</th>
      <td>N.Y.C.</td>
      <td>61761</td>
      <td>16.17</td>
      <td>1.35</td>
      <td>-1.72</td>
    </tr>
  </tbody>
</table>
</div>



Looks like the data span the year in time from May 1, 2019 through the end of April 2020, as intended.

# Run the simulation
In this section, we'll define a function, `simulate_battery`, that simulates the operation of the battery for energy arbitrage over the course of a year. Here are the inputs to the function:

- `initial_level`, the initial level of battery charge at start of simulation (kWh)
- `price_data`, the `DataFrame` with the hourly LBMP (\$/MWh)
- `max_discharge_power_capacity`, $$\kappa$$ (kW)
- `max_charge_power_capacity`, also $$\kappa$$ (kW)
- `discharge_energy_capacity` (kWh)
- `efficiency`, the AC-AC Round-trip efficiency, $$\eta$$ (unitless)
- `max_daily_discharged_throughput`, $$\tau$$ (kWh)
- `time_horizon`, the optimization time horizon (h), assumed here to be greater than or equal to 24.
- `start_day`, a pandas `Timestamp` for noon on the first simulation day

The function returns several outputs that can be used to examine system operation:

- `all_hourly_charges`, `all_hourly_discharges`, `all_hourly_state_of_energy`, charging and discharging activity, and state of energy, at an hourly time step (kWh)
- `all_daily_discharge_throughput`, discharged throughput at a daily time step (kWh)


```python
def simulate_battery(initial_level,
                     price_data,
                     max_discharge_power_capacity,
                     max_charge_power_capacity,
                     discharge_energy_capacity,
                     efficiency,
                     max_daily_discharged_throughput,
                     time_horizon,
                     start_day):
    #Track simulation time
    tic = time.time()
    
    #Initialize output variables
    all_hourly_charges = np.empty(0)
    all_hourly_discharges = np.empty(0)
    all_hourly_state_of_energy = np.empty(0)
    all_daily_discharge_throughput = np.empty(0)
    
    #Set up decision variables for optimization by
    #instantiating the Battery class
    battery = Battery(
        time_horizon=time_horizon,
        max_discharge_power_capacity=max_discharge_power_capacity,
        max_charge_power_capacity=max_charge_power_capacity)
    
    #############################################
    #Run the optimization for each day of the year.
    #############################################
    
    #There are 365 24-hour periods (noon to noon) in the simulation,
    #contained within 366 days
    for day_count in range(365):
        #print('Trying day {}'.format(day_count))
        
        #############################################
        ### Select data and simulate daily operation
        #############################################
        
        #Set up the 36 hour optimization horizon for this day by
        #adding to the first day/time of the simulation
        start_time = start_day \
        + pd.Timedelta(day_count, unit='days')
        end_time = start_time + pd.Timedelta(time_horizon-1, unit='hours')
        #print(start_time, end_time)
    
        #Retrieve the price data that will be used to calculate the
        #objective
        prices = \
        price_data[start_time:end_time]['LBMP ($/MWHr)'].values
                      
        #Create model and objective
        battery.set_objective(prices)

        #Set storage constraints
        battery.add_storage_constraints(
            efficiency=efficiency,
            min_capacity=0,
            discharge_energy_capacity=discharge_energy_capacity,
            initial_level=initial_level)
            
        #Set maximum discharge throughput constraint
        battery.add_throughput_constraints(
            max_daily_discharged_throughput=
            max_daily_discharged_throughput)

        #Solve the optimization problem and collect output
        battery.solve_model()
        hourly_charges, hourly_discharges = battery.collect_output()
        
        #############################################
        ### Manipulate daily output for data analysis
        #############################################
        
        #Collect daily discharge throughput
        daily_discharge_throughput = sum(hourly_discharges)
        #Calculate net hourly power flow (kW), needed for state of energy.
        #Charging needs to factor in efficiency, as not all charged power
        #is available for discharge.
        net_hourly_activity = (hourly_charges*efficiency) \
        - hourly_discharges
        #Cumulative changes in energy over time (kWh) from some baseline
        cumulative_hourly_activity = np.cumsum(net_hourly_activity)
        #Add the baseline for hourly state of energy during the next
        #time step (t2)
        state_of_energy_from_t2 = initial_level \
        + cumulative_hourly_activity
        
        #Append output
        all_hourly_charges = np.append(all_hourly_charges, hourly_charges)
        all_hourly_discharges = np.append(
            all_hourly_discharges, hourly_discharges)
        all_hourly_state_of_energy = \
        np.append(all_hourly_state_of_energy, state_of_energy_from_t2)
        all_daily_discharge_throughput = \
        np.append(
            all_daily_discharge_throughput, daily_discharge_throughput)
        
        #############################################
        ### Set up the next day
        #############################################
        
        #Initial level for next period is the end point of current period
        initial_level = state_of_energy_from_t2[-1]
        
        

    toc = time.time()
        
    print('Total simulation time: ' + str(toc-tic) + ' seconds')

    return all_hourly_charges, all_hourly_discharges, \
        all_hourly_state_of_energy,\
        all_daily_discharge_throughput
```

Now we'll run our simulation through the year, using the following illustrative values for the battery's parameters.


```python
max_discharge_power_capacity = 100 #(kW)
max_charge_power_capacity = 100 #(kW)
discharge_energy_capacity = 200 #(kWh)
efficiency = 0.85 #unitless
max_daily_discharged_throughput = 200  #(kWh)
```

To kick things off, we'll assume the battery is half-way charged to start.


```python
initial_level = discharge_energy_capacity/2
initial_level
```




    100.0




```python
all_hourly_charges, all_hourly_discharges, all_hourly_state_of_energy,\
all_daily_discharge_throughput = \
simulate_battery(initial_level=initial_level,
                 price_data=all_data,
                 max_discharge_power_capacity
                     =max_discharge_power_capacity,
                 max_charge_power_capacity
                     =max_charge_power_capacity,
                 discharge_energy_capacity=discharge_energy_capacity,
                 efficiency=efficiency,
                 max_daily_discharged_throughput
                     =max_daily_discharged_throughput,
                 time_horizon=36,
                 start_day=pd.Timestamp(
                     year=2019, month=5, day=1, hour=12,
                     tz='America/New_York'))
```

    Total simulation time: 20.976715087890625 seconds


Sanity check: the number of simulated hours should be:


```python
assert 24*365 == len(all_hourly_discharges)
```

# Analyze battery operation
Now we'll look at a suite of indicators of how the battery operated. We can check that all the constraints were satisfied, and analyze the financial impact of our system.

## Power output
Define power output with discharging as positive and charging as negative.


```python
mpl.rcParams["figure.figsize"] = [5,3]
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams.update({"font.size":12})
```


```python
plt.hist(all_hourly_discharges - all_hourly_charges)
plt.xlabel('kW')
plt.title('Hourly power output')
```




    Text(0.5, 1.0, 'Hourly power output')




![png](/assets/images/2020-05-03-energy-arbitrage/output_62_1.png)


This indicates that for most hours over the year, the power is close to zero. In other words, the battery is neither charging nor discharging. However it is also common for the battery to be operating at the limits of its range [-100, 100] kW.

## State of energy
The battery state of energy should be no less than zero, and no greater than the discharge energy capcacity, at any time: [0, 200] kWh.


```python
plt.hist(all_hourly_state_of_energy)
plt.xlabel('kWh')
plt.title('Hourly state of energy')
```




    Text(0.5, 1.0, 'Hourly state of energy')




![png](/assets/images/2020-05-03-energy-arbitrage/output_65_1.png)


Results indicate the battery is operating within the prescribed limits of state of energy.

## Revenue, cost, and profit

We'll analyze the following financial indicators:
- Total annual revenue generation (\$)
- Total annual charging cost (\$)

We'll also look at the total annual discharged throughput (kWh) later. To examine all of these, it is convenient to put the data in a `DataFrame`.

Select out a new `DataFrame` based on the time frame of simulation, to report further results:


```python
all_data_sim_time = all_data[
    pd.Timestamp(year=2019, month=5, day=1, hour=12, tz='America/New_York'):
    pd.Timestamp(year=2020, month=4, day=30, hour=11, tz='America/New_York')].copy()
```

Check there is the right number of rows:


```python
all_data_sim_time.shape
```




    (8760, 5)




```python
assert all_data_sim_time.shape[0] == len(all_hourly_discharges)
```

Attach simulation results


```python
#These indicate flows during the hour of the datetime index
all_data_sim_time['Charging power (kW)'] = all_hourly_charges
all_data_sim_time['Discharging power (kW)'] = all_hourly_discharges
all_data_sim_time['Power output (kW)'] = \
    all_hourly_discharges - all_hourly_charges
#This is the state of power at the beginning of the hour of the datetime index 
all_data_sim_time['State of Energy (kWh)'] = \
    np.append(initial_level, all_hourly_state_of_energy[0:-1])
```

Revenue and cost would be in units of
$$\frac{\text{kW} \cdot \text{\$} \cdot \text{h}}{\text{MWh}} = 1000 \cdot \$$$ , so divide by 1000 to adjust to \$:


```python
all_data_sim_time['Revenue generation ($)'] = \
all_data_sim_time['Discharging power (kW)'] \
* all_data_sim_time['LBMP ($/MWHr)'] / 1000
```


```python
all_data_sim_time['Charging cost ($)'] = \
all_data_sim_time['Charging power (kW)'] \
* all_data_sim_time['LBMP ($/MWHr)'] / 1000
```


```python
all_data_sim_time['Profit ($)'] = all_data_sim_time['Revenue generation ($)'] \
- all_data_sim_time['Charging cost ($)']
```

What is the total annual revenue generation?


```python
all_data_sim_time['Revenue generation ($)'].sum()
```




    2354.6574498602467



Total annual charging cost?


```python
all_data_sim_time['Charging cost ($)'].sum()
```




    1391.6754123382877



Calculate profit


```python
all_data_sim_time['Profit ($)'].sum()
```




    962.9820375219592



So we could make a profit of nearly \$963 by performing energy arbitrage.

## Total annual discharged throughput

How much energy has flowed through this battery during the course of the year? For some context here, the sum of daily discharged throughput is limited to 200 kWh/day. If the battery discharged its maximum possible energy every day during the 365 day simulation, the total discharge would be:


```python
365*200
#kWh
```




    73000



And in fact it is:


```python
sum(all_daily_discharge_throughput)
```




    72955.00000394997



This implies the system is hitting the maximum discharged throughput limit on most days. We can check this by doing a `value_counts()` on the `Series` of daily throughput.


```python
pd.Series(all_daily_discharge_throughput.round(0)).value_counts()
```




    200.0    364
    155.0      1
    dtype: int64



The battery operated at maximum throughput for all but one day.

## Find the most profitable week
Group the profit column by week and locate the maximum:


```python
max_profit_week = (all_data_sim_time['Profit ($)'].resample('W').sum() == \
all_data_sim_time['Profit ($)'].resample('W').sum().max()).values
```


```python
all_data_sim_time['Profit ($)'].resample('W').sum()[max_profit_week]
```




    Time Stamp
    2019-07-21 00:00:00-04:00    51.015471
    Freq: W-SUN, Name: Profit ($), dtype: float64



A week in July was the most profitable for energy arbitrage. For this week, let's make a graph of hourly battery state of energy and hourly LBMP.


```python
mpl.rcParams["figure.figsize"] = [8,6]
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams.update({"font.size":14})
```


```python
most_profit_week_start = pd.Timestamp(
    year=2019, month=7, day=21, tz='America/New_York')
ax = all_data_sim_time[
    most_profit_week_start:most_profit_week_start+pd.Timedelta(weeks=1)]\
[['State of Energy (kWh)', 'LBMP ($/MWHr)']]\
.plot(secondary_y='LBMP ($/MWHr)', mark_right=False)

ax.set_ylabel('State of energy (kWh)')
ax.right_ax.set_ylabel('LBMP ($/MWh)')
ax.get_legend().set_bbox_to_anchor((0.3, 1))
```


![png](/assets/images/2020-05-03-energy-arbitrage/output_99_0.png)


The battery appears to follow the general arbitrage srategy of "buy low, sell high", or in this case "charge cheaply, discharge discreetly" to take advantage of future price changes. It may be that during this week, it was quite warm in New York City, leading to high power demand for air conditioning, large price swings between day and night, and a good opportunity for our battery system to make some money.

## Monthly profit


```python
mpl.rcParams["figure.figsize"] = [6,4]
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams.update({"font.size":12})
```


```python
all_data_sim_time['Profit ($)'].resample('M').sum().plot()
plt.ylabel('Total monthly profit ($)')
```




    Text(0, 0.5, 'Total monthly profit ($)')




![png](/assets/images/2020-05-03-energy-arbitrage/output_103_1.png)


Profit is mostly higher in the summer than winter, but is also high in January and December. A plot of the LBMP throughout the year sheds some light on this:


```python
all_data_sim_time['LBMP ($/MWHr)'].plot()
plt.ylabel('LBMP ($/MWHr)')
```




    Text(0, 0.5, 'LBMP ($/MWHr)')




![png](/assets/images/2020-05-03-energy-arbitrage/output_105_1.png)


Changes in price enable the arbitrage strategy to make profit. Generally, most of the larger price swings are during the summer months, probably reflecting increased demand due to air conditioning usage on hot summer days. But there are large price changes in November, December, and January. This may be due to tourism in New York City during the holiday season, or cold weather that increases demand for electricity for heating.

# Conclusions

We found that an energy arbitrage strategy for a grid-connected battery can be formulated using linear programming, assuming future prices are known over some time horizon. We showed that when operating under an illustrative set of system parameters and using real-world energy price data, such a system can generate an annual profit of \$963.

Further optimization for increased profit may be possible, if prices are able to be accurately predicted beyond the 36 hour optimization horizon used here. The NYISO price determination involves a load forecasting model, that depends on economic and weather factors. It may be possible to include such factors in a price forecasting model to estimate future day-ahead market prices that are not yet public. In another interesting direction, Wang and Zhang (2018) show that reinforcement learning using historical price data can lead to higher profits than maximizing instantaneous profit, suggesting other possible approaches to maximizing profit from energy arbitrage.

I hope you found this post helpful for understanding how linear programming can be used to formulate an optimal arbitrage strategy if future prices are known.

### References

All references were accessed on May 2, 2020.

---

NYISO. [Day-Ahead Scheduling Manual](https://www.nyiso.com/documents/20142/2923301/dayahd_schd_mnl.pdf/0024bc71-4dd9-fa80-a816-f9f3e26ea53a).

PJM Interconnection LLC. [Locational Marginal Pricing Components](https://www.pjm.com/-/media/training/nerc-certifications/markets-exam-materials/mkt-optimization-wkshp/locational-marginal-pricing-components.ashx?la=en).

Salles, Mauricio B. C., et al. 2017. [Potential Arbitrage Revenue of Energy Storage Systems in PJM](https://www.mdpi.com/1996-1073/10/8/1100/htm). Energies 10:8.

Sioshansi, Ramteen, et al. 2009. [Estimating the Value of Electricity Storage in PJM: Arbitrage and Some Welfare Effects](https://www.sciencedirect.com/science/article/pii/S0140988308001631). Energy Economics 31:2, 269-277.

Wang, Hao and Zhang, Baosen, 2018. [Energy Storage Arbitrage in Real-Time Markets via Reinforcement Learning](https://arxiv.org/abs/1711.03127). IEEE PES General Meeting.

---

I found this [guide to getting started with linear programming in PuLP](https://benalexkeen.com/linear-programming-with-python-and-pulp/), by Ben Alex Keen, to be very helpful.
