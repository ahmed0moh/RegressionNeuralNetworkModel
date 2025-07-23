# 🤖 Regression Neural Network with Keras

This Jupyter Notebook project is a hands-on practice in *deep learning* and regression using *Keras* (TensorFlow). We build a feed-forward neural network to predict concrete strength from compositional features.

---

## 📚 Introduction

We use the “Concrete Compressive Strength” dataset, which includes features such as Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, and Fine Aggregate.  
Our goal is to predict the concrete’s *compressive strength* (MPa) using a neural network regression model.

---

## 🛠 Libraries Used

- *pandas* – data loading & manipulation  
- *numpy* – numerical operations  
- *matplotlib* – plotting  
- *tensorflow.keras* – building & training the neural network  
- *scikit-learn* – train/test split & metrics  

---

## 🚦 Steps Overview

1. *Load & Inspect Data*  
2. *Preprocess*: handle missing values, visualize distributions, normalize features  
3. *Define Model*: build a Sequential Keras model with Dense layers  
4. *Train & Evaluate*: fit on training data, compute MSE over multiple random splits  
5. *Visualize Results*: plot data histograms and MSE vs. epoch  

---

## 1. Load & Inspect Data

```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

concrete_data = pd.read_csv('concrete_data.csv')
concrete_data.head(10)
concrete_data.isnull().sum()
```

---

2. Exploratory Data Analysis
```python
concrete_data.hist(bins=10, figsize=(12,10))
plt.tight_layout()
plt.show()
```

![Feature Distributions](https://github.com/ahmed0moh/RegressionNeuralNetworkModel/blob/main/Plots/concrete_data_hist.png)


---

3. Preprocessing
```python
# Separate features and target
X = concrete_data.drop("Strength", axis=1)
y = concrete_data["Strength"]

# Normalize features
X_norm = (X - X.mean()) / X.std()
X_norm.head()
```

---

4. Model Definition
```python
def create_model():
    model = Sequential([
        Dense(10, activation='relu', input_shape=(X_norm.shape[1],)),
        Dense(10, activation='relu'),
        Dense(10, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1)  # output layer
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
```

---

5. Training & Evaluation

We run 50 random train/test splits, train for 100 epochs each, and record the MSE:
```python
mse_list = []
for _ in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3)
    model = create_model()
    model.fit(X_train, y_train, epochs=100, verbose=0)
    y_pred = model.predict(X_test)
    mse_list.append(mean_squared_error(y_test, y_pred))

mean_mse = np.mean(mse_list)
std_mse  = np.std(mse_list)
print(f"Mean MSE: {mean_mse:.4f}")
print(f"Std  MSE: {std_mse:.4f}")
```

---

6. Results & Plots

Mean MSE: {{ mean_mse }}

Std  MSE: {{ std_mse }}


Plot MSE vs. iteration:
```python
plt.plot(mse_list)
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.title('MSE over 50 Splits')
plt.show()
```


![MSE over Iterations](https://github.com/ahmed0moh/RegressionNeuralNetworkModel/blob/main/Plots/MSE_vs_Epochs.png)


---

▶ How to Run

1. Clone the repository.


2. Open regression_neural_network.ipynb in Jupyter.


3. Ensure concrete_data.csv is in the same folder.


4. Run all cells sequentially.




---

📂 Files
```
.
├── regression_neural_network.ipynb   # Jupyter Notebook
├── concrete_data.csv                 # Dataset
└── README.md                         # This file
```

---

📖 References

Keras Docs – Sequential model

Dataset source – UCI Machine Learning Repository


---

🙋 Author

Ahmed Dakrory — practicing deep learning with Keras.
