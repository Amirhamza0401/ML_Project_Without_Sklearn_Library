"""Univariate linear Regression"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Take input
x = []
y = []
n = int(input("Enter the number of data points: "))

for i in range(n):
    a = float(input(f"Enter x element {i+1}: "))
    x.append(a)

for i in range(n):
    b = float(input(f"Enter y element {i+1}: "))
    y.append(b)

df = pd.DataFrame({'x': x, 'y': y})
print(df)

alpha = float(input("Enter the value of alpha: "))
previous_J = float('inf')
t0 = 0
t1 = 0
b = 0.0000001

while True:
    t = 0
    dt0 = 0
    dt1 = 0
    for i in range(n):
        h = t1 * x[i] + t0
        a = h - y[i]
        t += (a ** 2)
        dt0 += (h - y[i])
        dt1 += (h - y[i]) * x[i]
    J = t / (2 * n)
    if abs(previous_J - J) < b:
        break
    previous_J = J
    t0 = t0 - alpha * dt0 / n
    t1 = t1 - alpha * dt1 / n

y_pred = [t1 * xi + t0 for xi in x]

print(f"Intercept (t0): {t0}")
print(f"Coefficient (t1): {t1}")
print(f"Final Regression Equation: y = {t1} * x + {t0}")

plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.show()

# Predict for a new value
x_new = float(input("Enter a new x value to predict y: "))
y_new = t1 * x_new + t0
print(f"Predicted y for x = {x_new} is: {y_new}")





"""Enter the number of data points: 20
Enter x element 1: 1
Enter x element 2: 2
Enter x element 3: 3
Enter x element 4: 4
Enter x element 5: 5
Enter x element 6: 6
Enter x element 7: 7
Enter x element 8: 8
Enter x element 9: 9
Enter x element 10: 10
Enter x element 11: 11
Enter x element 12: 12
Enter x element 13: 13
Enter x element 14: 14
Enter x element 15: 15
Enter x element 16: 16
Enter x element 17: 17
Enter x element 18: 18
Enter x element 19: 19
Enter x element 20: 20
Enter y element 1: 2.3
Enter y element 2: 4.1
Enter y element 3: 6.2
Enter y element 4: 8.1
Enter y element 5: 9.9
Enter y element 6: 12.2
Enter y element 7: 14.1
Enter y element 8: 16.3
Enter y element 9: 18.1
Enter y element 10: 20.5
Enter y element 11: 22.4
Enter y element 12: 24.2
Enter y element 13: 26.1
Enter y element 14: 28.4
Enter y element 15: 30.3
Enter y element 16: 32.1
Enter y element 17: 34.5
Enter y element 18: 36.2
Enter y element 19: 38
Enter y element 20: 40.4
       x     y
0    1.0   2.3
1    2.0   4.1
2    3.0   6.2
3    4.0   8.1
4    5.0   9.9
5    6.0  12.2
6    7.0  14.1
7    8.0  16.3
8    9.0  18.1
9   10.0  20.5
10  11.0  22.4
11  12.0  24.2
12  13.0  26.1
13  14.0  28.4
14  15.0  30.3
15  16.0  32.1
16  17.0  34.5
17  18.0  36.2
18  19.0  38.0
19  20.0  40.4
Enter the value of alpha: .01
Intercept (t0): 0.14694188815447487
Coefficient (t1): 2.0070624439564604
Final Regression Equation: y = 2.0070624439564604 * x + 0.14694188815447487

Enter a new x value to predict y: 25
Predicted y for x = 25.0 is: 50.32350298706598"""