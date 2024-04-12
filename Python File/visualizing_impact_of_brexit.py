import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Get the dataset
dataset = joblib.load("Dataset")
dataset = dataset[["TDate", "Price"]]
dataset.info()

# Extract time features
dataset["Year"] = pd.to_datetime(dataset["TDate"]).dt.year 
dataset["Month"] = pd.to_datetime(dataset["TDate"]).dt.month
dataset["Day"] = pd.to_datetime(dataset["TDate"]).dt.day


# Analyzing yearly average house prices during UK Brexit transition
year_data = dataset
year_data = year_data.drop(["TDate", "Month", "Day"], axis = 1)
year_data = year_data.groupby("Year").mean(numeric_only = True).reset_index()

segment_ranges = [(2010, 2016), (2016, 2020), (2020, 2023)]
colors = ['blue', 'red', 'green']
info = ["Before Referendum", "During Negotiations for Brexit", "Brexit Begins"]

plt.figure(figsize = (35, 10))
plt.title("Analyzing Yearly Average Prices in UK")
plt.plot(year_data["Year"], year_data["Price"], "--g", marker = 'o', markersize = 10, alpha = 0.5)
plt.axvline(x = 2016, color='cyan', linestyle='--', linewidth=2, alpha = 0.9, label = "Referendum for Brexit")
plt.axvline(x = 2020, color='cyan', linestyle='--', linewidth=2, alpha = 0.9, label = "UK leaves European Union(EU)")
for i, (start, end) in enumerate(segment_ranges):
    segment_data = year_data[(year_data["Year"] >= start) & (year_data["Year"] <= end)]
    plt.plot(segment_data["Year"], segment_data["Price"], marker='o', markersize=10, color=colors[i], label = info[i])
plt.xlabel("Year", labelpad = 20)
plt.ylabel("Average Price", labelpad = 20)
plt.legend()
plt.show()



# Analyzing monthly average house prices during UK Brexit transition
month_data = dataset
month_data["Year-Month"] = month_data["Year"].astype(str) + "-" + month_data["Month"].astype(str)
month_data = month_data.drop(["TDate", "Month", "Year", "Day"], axis = 1)
month_data = month_data.groupby("Year-Month").mean(numeric_only = True).reset_index()

segment_ranges1 = [("2010-1", "2016-1"), ("2016-1", "2020-2"), ("2020-2", "2023-4")]
colors = ['blue', 'red', 'green']

plt.figure(figsize = (45, 10))
plt.title("Analyzing Monthly Average Prices in UK")
plt.plot(month_data["Year-Month"], month_data["Price"], "--g", marker = 'o', markersize = 5, alpha = 0.5)
plt.axvline(x = "2016-1", color='cyan', linestyle='--', linewidth=2, alpha = 0.9, label = "Referendum for Brexit")
plt.axvline(x = "2020-2", color='cyan', linestyle='--', linewidth=2, alpha = 0.9, label = "UK leaves European Union(EU)")
for i, (start, end) in enumerate(segment_ranges1):
    segment_data1 = month_data[(month_data["Year-Month"] >= start) & (month_data["Year-Month"] <= end)]
    plt.plot(segment_data1["Year-Month"], segment_data1["Price"], marker='o', markersize=10, color=colors[i], label = info[i])
plt.xticks(rotation = 90, ha ='right')
plt.xlabel("Months", labelpad = 20)
plt.ylabel("Average Price", labelpad = 20)
plt.legend()
plt.show()



# Analyzing maximum house prices before referendum for brexit, during the transition and after brexit
before_referendum_brexit = round(year_data[year_data["Year"] <= 2016]["Price"].max(), 2)
during_negotiations_brexit = round(year_data[(year_data["Year"] > 2016) & (year_data["Year"] <= 2020)]["Price"].max())
after_brexit = round(year_data[year_data["Year"] > 2020]["Price"].max())
y = [before_referendum_brexit, during_negotiations_brexit, after_brexit]
X = ["Before Referendum", "During Negotiations for Brexit", "Brexit Begins"]

plt.figure(figsize = (15, 10))
plt.title("Analyzing Maximum Yearly Average House Prices in UK during Brexit Transition")
container = plt.bar(X, y, width = 0.5, color = colors, alpha = 0.5)
plt.bar_label(container, labels = y, padding = 10)
plt.xlabel("Brexit", labelpad = 20)
plt.ylabel("Maximum House Price", labelpad = 20)
plt.show()



# Analyzing house prices before referendum for brexit, during the transition and after brexit
before_referendum_brexit_min = round(year_data[year_data["Year"] <= 2016]["Price"].min(), 2)
during_negotiations_brexit_min = round(year_data[(year_data["Year"] > 2016) & (year_data["Year"] <= 2020)]["Price"].min())
after_brexit_min = round(year_data[year_data["Year"] > 2020]["Price"].min())
y_min = [before_referendum_brexit_min, during_negotiations_brexit_min, after_brexit_min]
X_min = ["Before Referendum", "During Negotiations for Brexit", "Brexit Begins"]

plt.figure(figsize = (15, 10))
plt.title("Analyzing Minimum Yearly Average House Prices in UK during Brexit Transition")
container_min = plt.bar(X_min, y_min, width = 0.5, color = colors, alpha = 0.5)
plt.bar_label(container_min, labels = y_min, padding = 10)
plt.xlabel("Brexit", labelpad = 20)
plt.ylabel("Minimum House Price", labelpad = 20)
plt.show()


"""
CONCLUSION

Brexit was a turning point in the history of the United Kingdom. It marked the day the 
United Kingdom broke away from European Union. On the 23rd of June, 2016, a referendum
was held to decide the position of the United Kingdom in realtion to Brexit. The transition
and withdrawal from the European Union was one that went on for years. On the 31st of January,
2020, the brexit agreements were finalised and came into force.

We attempt understanding the impact of Brexit on house prices in the United Kingdom. This 
relationship is analysed using 4 approaches:
    - Average yearly house prices in the United Kingdom
    - Average monthly house prices in the United Kingdom
    - Average maximum yearly house prices in the United Kingdom
    - Average minimum yearly house prices in the United Kingdom
    
The average yearly house prices in the UK gives an overall idea and glance into how house prices
fluctuate in 3 periods, these are: before brexit, during negotiations, and after brexit. Across
these periods we analysed, we notice a trend of prices generally peaking each year. Two significant
points worth noticing were between the year 2016, the year negotiations for Brexit began. We see a 
spike in average house prices between then and 2017. This period recorded the highest shift in 
yearly average prices in the UK, going from 309,720 in 2016 to 346,982 in 2017. The other period 
worth noticing is the peak yearly average house price in UK which occured in 2022, the figure 
amounting to 393,173.

The average monthly house prices in the UK gives a clearer picture of the fluctuations in house 
prices before brexit, during negotiations, and after brexit. The key take away from this analysis
is the stability in the prices across different years before brexit compared to during negotiations,
and after brexit. We see slight increase in the average house prices in the years before brexit, 
however, this increase has little oscillation across different months. During negotiations and the
transition towards brexit, we start seeing strong oscillation in the average monthly house prices. 
The stablilty which we could see in our graph before brexit dwindles. The period after brexit 
sees the yearly average house prices in the UK reach it's peak while witnessing stronger 
oscillation of the prices. This flunctuations witnessed could be due to uncertainty, interest rates,
affected trade relations, and other factors that the United Kingdom has had to deal with after brexit.

In our bar chart, we analyse the maximum and minimum yearly average house prices in the United Kingdom. 
Our analysis focuses on the 3 periods previously specified. We see the following results:
    - Average maximum yearly house prices in the United Kingdom:
        * Before Brexit - 309,720
        * During Transition - 365,597
        * After Brexit - 393,173
    
    - Average minimum yearly house prices in the United Kingdom:
        * Before Brexit - 231,631
        * During Transition - 346,982
        * After Brexit - 352,649
From our analysis we see the periods before brexit have the minimum average yearly house prices in the
United Kingdom, while the period after brexit has the maximum average yearly house pricesin the United
Kingdom. The difference in the average maximum and minimum yearly house prices in the United Kingdom between the 
period before and after brexit illustrates the consisitent rise in house prices and the big impact brexit
played in house price increase.
"""
