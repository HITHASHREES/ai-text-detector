import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.random.rand(200, 768)
y = np.array([0]*100 + [1]*100)

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

joblib.dump(model, "model.joblib")
print("Model saved")
