import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
import pickle

df = pd.read_csv("area.csv")
x = df[['area']]
y = df.price

# print(df)
# print(x)
# print(y)

plt.title("Area and Price graaph")
plt.xlabel("Area")
plt.ylabel("Price")
plt.scatter(x, y, color="red")

# plt.show()

model = LR()
model.fit(x, y)
value = int(input("Enter the area: "))
value = model.predict([[value]])[0]

# print("Price is", value)

pickle.dump(model, open("areaPrice.pkl", "wb"))