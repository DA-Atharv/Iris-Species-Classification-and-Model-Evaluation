# Iris-Species-Classification:

## Introduction:
The Iris flower data set or Fisher’s Iris data set is a multivariate data set introduced by the British statistician, eugenicist, and biologist Ronald Fisher in his 1936 paper the use of multiple measurements in taxonomic problems as an example of linear discriminant analysis.

## Data Collection:
The dataset is collected from Kaggle[https://www.kaggle.com/uciml/iris].
This dataset consists of 3 categories of species which is setosa, versicolor and virginica.
We can find two kind of data from kaggle which is CSV data and SQLITE database.
Each iris species consists of 50 samples.
The features of iris flower are Sepal Length in cm, Sepal Width in cm, Petal Length in cm and Petal Width in cm.
![image](https://github.com/DA-Atharv/Iris_Species_Classification/assets/159448408/3a451bf4-c937-4b84-b059-1074151aa112)

## Data Processing:
- Loaded the data.
```python
# Load Iris csv dataset
iris_data = pd.read_csv('../data/iris.csv')
```

## Exploratory Data Analysis (EDA):
- Let’s group the data by species and do some descriptive statistics
```python
# Groupby Species for descriptive statistics
iris_data.groupby('species').describe().T
```
![image](https://github.com/DA-Atharv/Iris_Species_Classification/assets/159448408/ebcf5c1c-49d4-4f90-a526-838bce3ce054)
- **count** shows that there 50 samples for each species.
- **Setosa**
  - Average sepal length is 5cm
  - Average sepal width is 3cm
  - Average petal length is 1.5cm
  - Average petal width is 0.25cm
- **Versicolor**
  - Average sepal length is 6cm
  - Average sepal width is 2.8cm
  - Average petal length is 4.26cm
  - Average petal width is 1.32cm
- **Virginica**
  - Average sepal length is 6.6cm
  - Average sepal width is 3cm
  - Average petal length is 6cm
  - Average petal width is 2cm

- From the above information,
  - Based on Petal length we can easily classify them as *Setosa(1.5cm), Versicolor(4.2cm) and Virginica(6cm)*.
  - Based on Petal width we can easily classify *Setosa(0.25cm) from Versicolor(1.32cm) and Virginica(2cm)*.
  - Sepal width looks similar for all three species — *Setosa(3cm), Versicolor(2.8cm) and Virginica(3cm)*.
  - Based on Sepal length, there are only small changes on three species (5cm, 6cm and 6.6cm)
Since Sepal width looks similar for all the species, we can drop that feature.

## Feature Observations:
![image](https://github.com/DA-Atharv/Iris-Species-Prediction-Model-Evaluation/assets/159448408/8b2d33e5-dfec-4ac3-ba64-4ab3120c8669)

## Analysis:
From the mathematical models i used i can confirm that using petal features gives more accuracy.
Further it was validated by the heatmap high correlation between petal length and width than that of sepal length and width. 
