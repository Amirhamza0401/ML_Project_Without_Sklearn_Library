"""Multivariate"""
import pandas as pd
import matplotlib.pyplot as plt
n = (input("Enter the number of samples: "))
m = int(input("Enter the number of features: "))
y=[]
d = []
for i in range(n):
    d1 = [1]
    for j in range(m):
        val = float(input(f"Enter element of {i+1},{j+1}: "))
        d1.append(val)

    d.append(d1)
    Y = float(input(f"Enter the output y{i+1}: "))
    y.append(Y)

x = pd.DataFrame(d, columns=[f"x{j}" for j in range(m+1)])
x["y"] = y
print(x)
previous_J=float('inf')
alpha=float(input("enter the alpha:"))

t=[0.0 for i in range(m+1)]

b=0.000000001
while True:
    dt=[0.0 for i in range(m+1)]
    t1=0
    for i in range (n):
        xi = x.iloc[i, :-1].values # it will take all column except last column
        h =sum(t[j] * xi[j] for j in range(m+1))
        a=h-y[i]
        t1+=(a**2)
        for j in range(m+1):
            dt[j] += a * xi[j]

    J=t1/(2*n)
    if abs(previous_J-J)<b:
        break
    previous_J=J
    for j in range(m+1):
       t[j] = t[j] - alpha * dt[j] / n
print(t)

Xi = x.iloc[:, :-1].values

y_pred = []

for xi in Xi:
    pred = 0
    for j in range(m+1):
        pred += t[j] * xi[j]
    y_pred.append(pred)
plt.scatter(range(n), y, label="Actual y", marker='o')
plt.plot(range(n), y_pred, label="Predicted y", marker='x')
plt.xlabel("Sample Index")
plt.ylabel("Output")
plt.title("Actual vs Predicted Output")
plt.legend()
plt.grid(True)
plt.show()
# Add predicted values as a new column
x["y_pred"] = y_pred
print("\nDataset with Actual and Predicted y values:")
print(x)

"""
x_new = [1]  # start with x0 =1

for j in range(m):
    val = float(input(f"Enter value for feature x{j+1}: "))
    x_new.append(val)

# Step 2: Predict y
y_new = sum(t[j] * x_new[j] for j in range(m+1))

# Step 3: Print result
print(f"Predicted y for input {x_new[1:]} is: {y_new}")"""


"""Enter the number of samples: 10
Enter the number of features: 3
Enter element of 1,1: 1
Enter element of 1,2: 2
Enter element of 1,3: 3
Enter the output y1: 14.1
Enter element of 2,1: 2
Enter element of 2,2: 1
Enter element of 2,3: 4
Enter the output y2: 16.4
Enter element of 3,1: 3
Enter element of 3,2: 2
Enter element of 3,3: 5
Enter the output y3: 19.2
Enter element of 4,1: 4
Enter element of 4,2: 3
Enter element of 4,3: 2
Enter the output y4: 21.3
Enter element of 5,1: 5
Enter element of 5,2: 4
Enter element of 5,3: 3
Enter the output y5: 25
Enter element of 6,1: 6
Enter element of 6,2: 2
Enter element of 6,3: 6
Enter the output y6: 26.7
Enter element of 7,1: 7
Enter element of 7,2: 3
Enter element of 7,3: 5
Enter the output y7: 29.6
Enter element of 8,1: 8
Enter element of 8,2: 4
Enter element of 8,3: 6
Enter the output y8: 34.1
Enter element of 9,1: 9
Enter element of 9,2: 5
Enter element of 9,3: 4
Enter the output y9: 35.2
Enter element of 10,1: 10
Enter element of 10,2: 6
Enter element of 10,3: 5
Enter the output y10: 40
   x0    x1   x2   x3     y
0   1   1.0  2.0  3.0  14.1
1   1   2.0  1.0  4.0  16.4
2   1   3.0  2.0  5.0  19.2
3   1   4.0  3.0  2.0  21.3
4   1   5.0  4.0  3.0  25.0
5   1   6.0  2.0  6.0  26.7
6   1   7.0  3.0  5.0  29.6
7   1   8.0  4.0  6.0  34.1
8   1   9.0  5.0  4.0  35.2
9   1  10.0  6.0  5.0  40.0
enter the alpha:.01
[np.float64(7.6831785449790235), np.float64(2.183152870180875), np.float64(1.1523675526888277), np.float64(0.646885703819093)]


Dataset with Actual and Predicted y values:
   x0    x1   x2   x3     y     y_pred
0   1   1.0  2.0  3.0  14.1  14.111724
1   1   2.0  1.0  4.0  16.4  15.789395
2   1   3.0  2.0  5.0  19.2  19.771801
3   1   4.0  3.0  2.0  21.3  21.166664
4   1   5.0  4.0  3.0  25.0  25.149070
5   1   6.0  2.0  6.0  26.7  26.968145
6   1   7.0  3.0  5.0  29.6  29.656780
7   1   8.0  4.0  6.0  34.1  33.639186
8   1   9.0  5.0  4.0  35.2  35.680935
9   1  10.0  6.0  5.0  40.0  39.663341
\nx_new = [1]  # start with x0 =1\n\nfor j in range(m):\n    val = float(input(f"Enter value for feature x{j+1}: "))\n    x_new.append(val)\n\n# Step 2: Predict y\ny_new = sum(t[j] * x_new[j] for j in range(m+1))\n\n# Step 3: Print result\nprint(f"Predicted y for input {x_new[1:]} is: {y_new}")
"""