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
## Key Visualizations:
|                           |                           |
|---------------------------|---------------------------|
| Boxplot: It visually compares distributions of sepal length, sepal width, petal length, petal width based on numerical data through their quartiles. ![image](https://github.com/DA-Atharv/Iris-Species-Classification-and-Model-Evaluation/assets/159448408/c2b1caa8-5125-4850-bfd4-e39b1feeabe9) | Pairplot: Relationships between variables across multiple dimensions. ![image](https://github.com/DA-Atharv/Iris-Species-Classification-and-Model-Evaluation/assets/159448408/bd6c1aaf-3e87-4776-8394-cd84917d2ff4))
| Swarm-Plot: ![image](https://github.com/DA-Atharv/Iris-Species-Classification-and-Model-Evaluation/assets/159448408/640fdecb-81be-41ee-8b88-a997a080f828)) | (![image](https://github.com/DA-Atharv/Iris-Species-Classification-and-Model-Evaluation/assets/159448408/eec6d04d-05a1-4e5a-8570-882f0fde13ad)
| Voilin-Plot ![image](https://github.com/DA-Atharv/Iris-Species-Classification-and-Model-Evaluation/assets/159448408/d9ab3196-dc8d-449b-b628-4bcd3bd50741) | Cheking Corellation: Heatmap ![image](https://github.com/DA-Atharv/Iris_Species_Classification/assets/159448408/ebcf5c1c-49d4-4f90-a526-838bce3ce054) |
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

## Splitting the data into training and testing dataset:
```python
train, test = train_test_split(iris_data, test_size = 0.3) # dataset is split into 70% training and 30% testing
print(train.shape)
print(test.shape)
```
## use petal and sepalas features:
#### Training and testing data for petals and sepals:
```python
petal = iris_data[['petal_length','petal_width','species']]
sepal = iris_data[['sepal_length','sepal_width','species']]

#Iris_Petals:
train_p,test_p = train_test_split(petal, test_size=0.3, random_state=0) 
train_x_p = train_p[['petal_length','petal_width']]
train_y_p = train_p.species

test_x_p = test_p[['petal_length','petal_width']]
test_y_p = test_p.species

#Iris_Sepals:
train_s,test_s = train_test_split(sepal, test_size=0.3, random_state=0) #sepals
train_x_s = train_s[['sepal_length','sepal_width']]
train_y_s = train_s.species

test_x_s = test_s[['sepal_length','sepal_width']]
test_y_s = test_s.species
```
## Logistic Regression:
```python
model = LogisticRegression()
model.fit(train_x_p,train_y_p) 
prediction=model.predict(test_x_p) 
print('The accuracy of the Logistic Regression using Petals is:',metrics.accuracy_score(prediction,test_y_p))

model.fit(train_x_s,train_y_s) 
prediction=model.predict(test_x_s) 
print('The accuracy of the Logistic Regression using Sepals is:',metrics.accuracy_score(prediction,test_y_s))
```
+ The accuracy of the Logistic Regression using Petals is: 0.9777777777777777
+ The accuracy of the Logistic Regression using Sepals is: 0.8222222222222222
## Decision Tree: 
```python
model=DecisionTreeClassifier()
model.fit(train_x_p,train_y_p) 
prediction=model.predict(test_x_p) 
print('The accuracy of the Decision Tree using Petals is:',metrics.accuracy_score(prediction,test_y_p))

model.fit(train_x_s,train_y_s) 
prediction=model.predict(test_x_s) 
print('The accuracy of the Decision Tree using Sepals is:',metrics.accuracy_score(prediction,test_y_s))
```
+ The accuracy of the Decision Tree using Petals is: 0.9555555555555556
+ The accuracy of the Decision Tree using Sepals is: 0.6444444444444445
## Conclusion:
+ From the mathematical models i used i can confirm that using petal features gives more accuracy.
+ Further it was validated by the heatmap high correlation between petal length and width than that of sepal length and width. 
