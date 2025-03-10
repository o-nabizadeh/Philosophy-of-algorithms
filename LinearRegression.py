import numpy as np

#Example dataset
x = np.array([1,2,3,4,5])
y = np.array([1,3,4,5,6])

# step2 : Calculate mean of x and y
mean_x = np.mean(x)
mean_y = np.mean(y)

# step3 : Calculate coefficient (slope and intersept)
n = len(x)
numerator = 0
denominator = 0

for i in range(n):
    numerator += (x[i] - mean_x) * (y[i] - mean_y)
    denominator += (x[i] - mean_x) ** 2
    
beta_1 = numerator / denominator
beta_0 = mean_y - (beta_1 * mean_x)

#step4 : make prediction 
y_pred = beta_0 + beta_1 * x

#step5 : evaluate the model
mse = np.mean((y - y_pred) ** 2)
ss_total = np.sum((y - mean_y) ** 2)
ss_residual = np.sum((y - y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)

print(f'Mean squared error: {mse}')
print(f'R-squared : {r2}')
print(f'slope(beta_1) : {beta_1}')
print(f'Intercept(beta_0) : {beta_0}')