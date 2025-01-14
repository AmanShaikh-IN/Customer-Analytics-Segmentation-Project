# Customer Analytics Segmentation Project

## Problem Statement
To what extent does our platform’s acquisition channel influence the learning outcomes of our students?  
Are there any geographical locations where most of our students discover the platform, specifically through social media platforms like YouTube or Facebook?

## Project Description
This project uses real-world customer data to perform market segmentation—crucial for businesses to understand customer behavior and improve marketing efficiency. The project involves:

- **Data Preprocessing:** Handle missing data, standardize variables, and prepare the dataset.
- **Exploratory Data Analysis (EDA):** Analyze correlations and visualize data relationships.
- **Feature Engineering:** Create dummy variables for categorical data like acquisition channels and geographical regions.
- **Clustering Algorithms:**
  - **K-Means Clustering:** For segmenting customers into meaningful groups.
  - **Hierarchical Clustering:** To explore nested clusters.
- **Result Interpretation:** Analyze clustering outcomes to derive insights about customer behavior.

## Files in the Repository
- `customer_segmentation.py`: Python script containing the full code for preprocessing, analysis, clustering, and visualization.
- `data/customer_segmentation_data.csv`: Input data file used for analysis.
- `data/Segmentation data legend.xlsx`: Legend for all categorical data values.
- `outputs/`: Folder containing visualizations such as:
  - Correlation heatmap (`corr.png`)
  - Scatter plot of raw data (`scatter.png`)
  - Dendrogram for hierarchical clustering (`hierarchical.png`)
  - Elbow method line chart (`line_chart.png`)

## Requirements
Install the following Python libraries before running the script:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
