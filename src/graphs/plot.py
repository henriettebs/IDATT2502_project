import matplotlib.pyplot as plt

x = []
y = []
for line in open("src/graphs/main.txt", "r"):
    lines = [i for i in line.split()]
    x.append(lines[0])
    y.append(lines[1])

plt.title("Stock Prediction")
plt.xlabel('Date/Days')
plt.ylabel('Price')
plt.yticks(y)
plt.plot(x, y, marker = 'o', c = 'g')
  
plt.show()
