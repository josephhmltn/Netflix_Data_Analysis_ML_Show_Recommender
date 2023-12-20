# Netflix Data Analysis

## Data Overview and Initial Exploration 🌟
- **Dataset**: Contains Netflix titles with attributes like year, rating, duration, etc. Datasource: [Kaggle](https://www.kaggle.com/datasets/narayan63/netflix-popular-movies-dataset)
- **Initial Observation**: Identified data types and missing values.

## Data Cleaning and Transformation 🧼
- **Year and Votes Fields**: Extracted year and converted votes to numeric.
- **Dropped Missing Values**: Focused on rows with complete 'votes' and 'rating' data.
- **Result**: Cleaned dataset ready for analysis.

## Exploratory Data Analysis (EDA) 🔍

### Rating and Votes Distribution 📊
- **Histograms**: Illustrated distribution of IMDb ratings and vote counts.
- **Adjustments**: Utilized log scale for clarity in vote distribution.

![](images/ratings_dist.png)

### Genre Analysis 🎭
- **Bar Chart**: Showed frequency of top 10 genres.
- **Insight**: Identified most prevalent genres on Netflix.

![](images/genre_top10.png)

### Yearly Trends Analysis 📅
- **Line Graphs**: Analyzed trends in number of releases and average ratings (2000-2022).
- **Improved Legibility**: Adjusted x-axis labels for better clarity.

![](images/releases_per_year.png)

### Rating vs. Votes Correlation 📉
- **Scatterplot**: Explored the relationship between IMDb ratings and vote counts.
- **Finding**: A weak positive correlation observed.

![](images/scatplot_ratings_votes.png)

### Duration Analysis ⏳
- **Histogram**: Investigated distribution of show/movie durations.
- **Color Palette**: Consistently used 'Blues_r' as per preference.

![](images/duration_dist_minutes.png)

### Word Clouds 🌬️
- **Descriptions and Genres**: Created word clouds to visualize common words and genres.
- **Refinement**: Excluded filler words for more accurate representation.

![](images/wordcloud_description.png)

## Predictive Modeling: Recommendation System 🤖

### System Setup 🛠️
- **Approach**: Content-based recommendation using genre and description.
- **Feature Extraction**: Applied TF-IDF on combined text features.
- **Similarity Measure**: Calculated cosine similarity between titles.

### Function Testing and Validation 🔍
- **Functionality**: Tested with titles like "Breaking Bad", "Brooklyn Nine-Nine", and "Mindhunter".
- **Output**: Successfully generated lists of similar titles.

*Analysis conducted with a focus on data-driven insights and clarity.*
