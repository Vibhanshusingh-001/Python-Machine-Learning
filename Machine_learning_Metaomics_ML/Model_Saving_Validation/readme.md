# Model saving 

In machine learning, while working with **scikit-learn** library, we need to save the trained models in a file and restore them in order to reuse them to compare the model with other models, and to test the model on new data.  

- The saving of data is called **Serialization**.  
- Restoring the data is called **Deserialization**.  

Also, we deal with different types and sizes of data. Some datasets are easily trained (take less time to train) but datasets whose size is large (more than 1GB) can take a very large time to train on a local machine even with GPU.  

When we need the same trained data in some different project or later sometime, to avoid wastage of training time, we store the trained model so that it can be used anytime in the future.  

There are **two ways** we can save a model in scikit-learn:  

---

## Way 1: Pickle String  

The **pickle module** implements a fundamental, but powerful algorithm for serializing and de-serializing a Python object structure.  

Pickle model provides the following functions:  
- `pickle.dump` → to serialize an object hierarchy.  
- `pickle.load` → to deserialize a data stream.  

### Example: Apply K Nearest Neighbor on the iris dataset and then save the model.  

```python
from sklearn.neighbors import KNeighborsClassifier as KNN
import numpy as np

# Load dataset
from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=2018)

# import KNeighborsClassifier model
knn = KNN(n_neighbors=3)

# train model
knn.fit(X_train, y_train)
````

**Output:** Model trained.

---

### Save a model to string using pickle

```python
import pickle

# Save the trained model as a pickle string.
saved_model = pickle.dumps(knn)

# Load the pickled model
knn_from_pickle = pickle.loads(saved_model)

# Use the loaded pickled model to make predictions
knn_from_pickle.predict(X_test)
```

**Output:** Predictions generated using pickled model.

---

## Way 2: Pickled Model as a File Using Joblib

**Joblib** is the replacement of pickle as it is more efficient on objects that carry large numpy arrays.

These functions also accept file-like object instead of filenames.

* `joblib.dump` → to serialize an object hierarchy.
* `joblib.load` → to deserialize a data stream.

```python
from joblib import Parallel, delayed
import joblib

# Save the model as a pickle in a file
joblib.dump(knn, 'filename.pkl')

# Load the model from the file
knn_from_joblib = joblib.load('filename.pkl')

# Use the loaded model to make predictions
knn_from_joblib.predict(X_test)
```

**Output:** Predictions generated using joblib model.

```
```

