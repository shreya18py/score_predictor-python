import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Scores': [35, 40, 50, 55, 65, 70, 75, 85, 90]
}
df = pd.DataFrame(data)

# Model training
X = df[['Hours']]
y = df['Scores']
model = LinearRegression()
model.fit(X, y)

# Prediction
pred_hours = [[7.5]]
pred_score = model.predict(pred_hours)
print(f"Predicted score for 7.5 hours: {pred_score[0]:.2f}")

# Plot
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.title('Study Hours vs Scores')
plt.show()
