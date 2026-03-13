# Analyzing Buster Posey’s Career Using Python

## Introduction

Baseball has long been one of the most data-rich sports, and modern analytics relies heavily on publicly available datasets to evaluate player performance. One of the most widely used sources for historical baseball data is the **Lahman Baseball Database**, which contains detailed statistics for Major League Baseball players across more than a century.

For this project, I created a dataset analyzing the offensive career performance of **Buster Posey**, the former San Francisco Giants catcher and one of the most influential players of the Giants’ three World Series championship teams in **2010, 2012, and 2014**.

The goal of the project was to demonstrate how Python tools such as **pandas** and **NumPy** can be used to collect, clean, transform, and engineer features from real-world sports data. The final result is a structured dataset that can be used for statistical analysis, visualization, and sports analytics research.

---

# Motivating Question

The dataset was designed to explore several questions about Buster Posey’s career:

- How did Posey’s offensive performance evolve throughout his career?
- How did his performance compare to the **league average MLB player** each season?
- Did Posey perform differently in the **postseason compared to the regular season**?
- Did his performance change during the Giants’ **World Series championship seasons**?

Answering these questions requires combining multiple datasets, computing derived statistics, and comparing individual performance against league-wide benchmarks.

---

# Ethical Data Collection

The data used in this project comes from the **Lahman Baseball Database**, which is a publicly available dataset widely used in academic research, data science projects, and sports analytics.

Lahman Baseball Database  
https://www.seanlahman.com/baseball-archive/statistics/

Because the dataset is openly distributed for research and educational use, there are no scraping restrictions or privacy concerns associated with using the data. The project simply downloads and processes the data locally rather than collecting information directly from a website.

This approach ensures that the project follows responsible data use practices while allowing full transparency and reproducibility.

---

# Data Collection and Processing

The project dataset is constructed from two tables in the Lahman database:

- **Batting.csv** – regular season batting statistics for all MLB players  
- **BattingPost.csv** – postseason batting statistics

The process for building the dataset involves several steps.

### 1. Load the Raw Data

Both datasets are imported into Python using the pandas library.
batting = pd.read_csv("Batting.csv")
post = pd.read_csv("BattingPost.csv")


### 2. Filter for Buster Posey

Each MLB player has a unique identifier in the Lahman dataset.  
Buster Posey’s ID is:
poseybu01


The dataset is filtered to include only Posey’s seasons.

### 3. Clean and Standardize Variables

Column names are renamed to improve readability and consistency.

Examples include:

| Original | Renamed |
|--------|-------------|
| yearID | Season |
| G | Games |
| AB | At_Bats |
| HR | Home_Runs |
| BB | Walks |
| SO | Strikeouts |

This step makes the dataset easier to interpret and work with during analysis.

---

# Feature Engineering

After cleaning the raw data, the project generates several derived statistics to better measure offensive performance.

## Rate Statistics

Traditional baseball rate metrics were calculated, including:

- **Batting Average**
- **On Base Percentage**
- **Slugging Percentage**
- **OPS (On-base Plus Slugging)**

These statistics normalize player performance and allow comparisons across seasons.

---

## Per-Game Production

Because players do not always play the same number of games each season, additional metrics measure performance on a per-game basis.

Examples include:

- Hits_per_Game
- HR_per_Game
- RBI_per_Game
- Runs_per_Game
- Walks_per_Game
- Strikeouts_per_Game

These metrics help standardize production across seasons.

---

## League Average Benchmarks

To evaluate Posey’s performance relative to other players, the project calculates **league-wide seasonal averages** from the Lahman dataset.

Examples include:

- League_Hits_per_Game  
- League_HR_per_Game  
- League_RBI_per_Game  

From these values, **relative performance metrics** are created:

- Hits_vs_League  
- HR_vs_League  
- RBI_vs_League  

Positive values indicate seasons where Posey performed **above league average**.

---

## Postseason Performance

Postseason batting statistics from `BattingPost.csv` are aggregated by season to allow comparison with regular season performance.

Examples include:

- Post_Hits
- Post_HR
- Post_RBI
- Post_Games

Per-game postseason statistics are also calculated to make comparisons more meaningful.

---

## Regular Season vs Postseason Comparison

The dataset includes metrics measuring how Posey’s postseason production differed from his regular season baseline.

Examples include:

- HR_Post_vs_Reg
- Hits_Post_vs_Reg
- RBI_Post_vs_Reg

These metrics help evaluate whether Posey’s performance increased or decreased during playoff games.

---

# Championship Season Indicator

Buster Posey played a major role in the Giants’ three World Series championships:

- **2010**
- **2012**
- **2014**

The dataset includes a variable indicating these championship seasons so that analysts can compare performance during those years against the rest of his career.

---

# Final Dataset Overview

The project exports two datasets.

## Wide Format Dataset

**posey_master_wide.csv**

Each row represents one season of Posey’s career and includes:

- Raw batting statistics
- Rate statistics
- Per-game production metrics
- League average benchmarks
- League comparison metrics
- Postseason statistics
- Postseason comparison metrics

This format is useful for:

- regression analysis
- machine learning
- exploratory data analysis

---

## Long Format Dataset

**posey_master_long.csv**

This dataset was created using `pandas.melt()` and converts the data into long format.

Example structure:

| Season | Stat | Value |
|------|------|------|
| 2012 | HR_per_Game | 0.17 |

This structure is particularly useful for:

- Tableau dashboards
- grouped visualizations
- faceted statistical analysis

---

# Data Quality Considerations

Like most real-world datasets, this project required several data quality checks.

Important considerations include:

- **Missing postseason values** for seasons in which Posey did not appear in the playoffs
- **Differences in season length**, which required normalization using per-game statistics
- Potential **league environment changes** across years that affect league averages

These factors are important when interpreting the results of any statistical analysis using the dataset.

---

# Tools Used

This project was built using Python and the following libraries:

- pandas
- numpy

The dataset can also be used with visualization tools such as:

- Tableau
- matplotlib
- seaborn
- R (ggplot2)

---

# Code Repository

The full code used to construct the dataset is available here:

**GitHub Repository:**  
[(https://github.com/ethaniel2024/data-acquisition)]

The repository includes the Python script used to build the dataset, the original input data, and the final processed datasets.
