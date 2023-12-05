import matplotlib.pyplot as plt

data = []
for i in range(0, 10):
    string = ""
    with open(f'./data/Out{i}.txt', 'r') as file:
        string = file.read().replace('\n', '')
    data.append([float(x) for x in string.split()])

with open(f'./data/Ref.txt', 'r') as file:
    string = file.read().replace('\n', '')
    ref = [float(x) for x in string.split()]
    data.append(ref)

with open(f'./data/OutFin.txt', 'r') as file:
    string = file.read().replace('\n', '')
    fin = [float(x) for x in string.split()]
    data.append(fin)

# plot the data
for i in range(0, 12):
    plt.plot(data[i])
plt.show()
