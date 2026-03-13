# Buster Posey Career Performance Dataset
### Data Curation, Feature Engineering, and Baseball Analytics

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-orange)
![Dataset](https://img.shields.io/badge/Data-Lahman%20Baseball%20Database-green)
![Status](https://img.shields.io/badge/Project-Complete-success)

---

## Dataset Preview

| Season | HR_per_Game | HR_vs_League | Post_HR_per_Game |
|------|------|------|------|
| 2010 | 0.13 | 0.05 | 0.18 |
| 2012 | 0.15 | 0.07 | 0.12 |
| 2014 | 0.11 | 0.03 | 0.10 |

# Project Overview

This project curates a comprehensive dataset analyzing the career offensive performance of **Buster Posey**, former catcher for the **San Francisco Giants** and a central figure in the franchise's three World Series championships (2010, 2012, 2014).

Using the **Lahman Baseball Database**, the project integrates:

- Regular season batting statistics
- Postseason batting statistics
- League-wide seasonal benchmarks
- Derived performance metrics
- Comparative statistics measuring performance relative to league averages
- Postseason vs regular season performance differences

The goal of this project is to demonstrate **data acquisition, cleaning, transformation, and feature engineering** using Python while producing a dataset suitable for **data visualization, statistical analysis, and sports analytics research**.

---

# Key Questions Explored

This dataset supports analysis of several questions:

* How did Buster Posey's offensive production evolve throughout his career?
* How did Posey's performance compare to the **league average MLB player** each season?
* Did Posey perform differently in the **postseason compared to the regular season**?
* Did Posey's performance change during the **Giants' championship seasons**?

---

# Data Sources

The data used in this project comes from the **Lahman Baseball Database**, one of the most widely used public baseball datasets.

**Lahman Baseball Database**  
https://www.seanlahman.com/baseball-archive/statistics/

Two tables from the dataset are used:

### Batting.csv
Regular season batting statistics for all Major League Baseball players.

Key variables include:

| Variable | Description |
|--------|-------------|
| playerID | Unique player identifier |
| yearID | Season year |
| teamID | Team abbreviation |
| lgID | League |
| G | Games played |
| AB | At bats |
| H | Hits |
| HR | Home runs |
| RBI | Runs batted in |
| BB | Walks |
| SO | Strikeouts |
| 2B | Doubles |
| 3B | Triples |

---

### BattingPost.csv

Contains postseason batting statistics including performance during:

- Division Series
- League Championship Series
- World Series

Postseason data is aggregated by **season** to allow comparison with regular season performance.

---

# Data Processing Pipeline

The dataset is constructed through the following steps.

## 1. Load Raw Data

The Lahman datasets are loaded using pandas.

```python
batting = pd.read_csv("Batting.csv")
post = pd.read_csv("BattingPost.csv")

## 2. Filter to Buster Posey

Buster Posey's Lahman player identifier is:
poseybu01


The dataset is filtered to include only Posey's career seasons.

## 3. Data Cleaning

Columns are renamed to improve readability.

| Original | Renamed |
|--------|-------------|
| yearID | Season |
| G | Games |
| AB | At_Bats |
| HR | Home_Runs |
| BB | Walks |
| SO | Strikeouts |

## 4. Feature Engineering

Several new statistics are calculated.

### Traditional Rate Statistics

**Batting Average**
Hits / At_Bats

**On Base Percentage**
(Hits + Walks + Hit_By_Pitch) / (At_Bats + Walks + Hit_By_Pitch + Sacrifice_Flies)

**Slugging Percentage**
Total_Bases / At_Bats

**OPS**
On_Base_Percentage + Slugging_Percentage

### Per-Game Production
To normalize production across seasons with different numbers of games, the dataset includes:
Hits_per_Game
HR_per_Game
RBI_per_Game
Runs_per_Game
Walks_per_Game
Strikeouts_per_Game

## 5. League Average Benchmarks
League-wide averages are calculated from the Lahman dataset.
Example metrics include:
League_Hits_per_Game
League_HR_per_Game
League_RBI_per_Game


From these values, **relative performance metrics** are calculated:
Hits_vs_League
HR_vs_League
RBI_vs_League


Positive values indicate Posey performed **above league average**.

## 6. Postseason Performance
Postseason statistics are aggregated by season.
Example postseason metrics include:
Post_Hits
Post_HR
Post_RBI
Post_Games

Postseason per-game metrics are also calculated:
Post_Hits_per_Game
Post_HR_per_Game

## 7. Regular Season vs Postseason Comparisons
The dataset includes variables measuring differences between postseason and regular season performance.
Examples:
Hits_Post_vs_Reg
HR_Post_vs_Reg
RBI_Post_vs_Reg

These metrics measure whether Posey performed **better or worse in the postseason** relative to his regular season baseline.

## 8. Championship Season Indicator
Posey's three World Series championship seasons are flagged:

2010  
2012  
2014  

This allows comparison of performance during championship runs.

---

# Dataset Outputs

The script exports two datasets.

---

## posey_master_wide.csv

Each row represents one season of Posey's career.

Columns include:

- Raw counting statistics  
- Rate statistics  
- Per-game metrics  
- League average statistics  
- Relative performance metrics  
- Postseason statistics  
- Postseason comparison metrics  

This dataset is well suited for:

- statistical modeling  
- regression analysis  
- machine learning  
- exploratory data analysis  

## posey_master_long.csv

A long-format dataset created using `pandas.melt()`.

### Structure

| Season | Stat | Value | Stat_Type |
|------|------|------|------|

### Example

| Season | Stat | Value |
|------|------|------|
| 2012 | HR_per_Game | 0.17 |

This format is ideal for:

- Tableau dashboards  
- faceted visualizations  
- grouped statistical analysis  

---

# Repository Structure

buster-posey-dataset  
│  
├── data  
│   ├── Batting.csv  
│   ├── BattingPost.csv  
│  
├── scripts  
│   └── build_posey_dataset.py  
│  
├── output  
│   ├── posey_master_wide.csv  
│   ├── posey_master_long.csv  
│  
├── README.md  

---

# Tools Used

Python libraries used in this project:

- pandas  
- numpy  

Optional tools for analysis and visualization:

- Tableau  
- Python (matplotlib, seaborn)  
- R (ggplot2)  

# Example Visualizations

The dataset supports several analytical visualizations:

- Career production trends  
- League-relative performance comparisons  
- Distribution of offensive metrics  
- Postseason vs regular season comparisons  

Example charts might include:  
Time Series: HR_per_Game vs League_HR_per_Game  
Histogram: Distribution of Hits_per_Game  
Postseason Comparison: HR_Post_vs_Reg  

# Author

Dataset curated as part of a data science coursework project demonstrating:

- Data acquisition  
- Data cleaning  
- Feature engineering  
- Dataset transformation  
- Preparation of analysis-ready data