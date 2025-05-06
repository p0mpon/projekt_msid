# Systems analysis and decision support methods - project

## Installation and running
You should have python and pip installed.

Create a virtual environment and install required libraries:
```bash
pip install -r requirements.txt
```

### Dataset
Download the dataset:  
[Student Performance Factors Dataset on Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)

### Running
To generate and save the results of data analysis and visualizations, run:
```bash
python data_analysis_main.py
```

# Part I

Data statistics and visualisations.

### Statistics include:
For numerical features:
- mean
- median
- minimum value
- maximum value
- standard deviation
- 5th percentile
- 95th percentile
- number of null values

For categorical features:
- number of unique values
- number of null values
- proportions ( {value}:{percentage}; )

### Visualizations include:
For numerical features:
- histograms
- distribution point error bars
- line charts comparing two features
- correlation of each two features

For categorical features:
- pie charts
- boxplots
- violinplots