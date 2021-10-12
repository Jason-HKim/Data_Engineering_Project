# Spotify Data Pipeline

## Abstract
My goal with this project was to utilize Spotify data to engineer a data pipeline, and deliver a product that allows data exploration and can predict song popularity.

## Design
In the music industry, artists and labels are constantly searching for ways to maximize artist exposure and music consumption. On spotify, this is quantified in the 'popularity' index, which takes into account number of listens and time. Songs with high numbers of current listens will have higher levels of popularity. But what features of a song make it popular?

## Data

I originally intended to use the Spotify API, and I plan to modify my code to be able to ingest updated data.

The dataset that I worked with was downloaded from [Kaggle](https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks), and consists of around 600,000 rows of track data, and 1.1 million rows of artist data.


## Tools & Algorithms
Python Libraries:
* SKLearn - Data Analysis & Modeling
* Pandas & Numpy - Data manipulation & cleaning
* MatPlotLib & Seaborn - Data visualizations
* Datetime - For dates
* Streamlit - For Deployment

Data Import:
* Download files from kaggle.com
* Imported files into DB Browser
* Stored tables into SQL database

## Communication

Heatmap of Features
![heatmap](https://github.com/Jason-HKim/Data_Engineering_Project/blob/master/Visualizations/heatmap.png)
