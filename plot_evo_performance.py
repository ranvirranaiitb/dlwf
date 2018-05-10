import matplotlib.pyplot as plt
import os

x = []
y = []
for filename in os.listdir('results'):
    with open('results/' + filename, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('BEST OF GEN'):
                if i < len(lines) - 1:
                    fitness = float(lines[i+1].split('=')[1].split('(')[0].strip())
                    if fitness > 2.81:
                        overhead = float(lines[i].split(',')[0].split('=')[1].strip())
                        acc = float(lines[i].split(',')[1].split('=')[1].strip())
                        x.append(overhead)
                        y.append(acc)

base_x = []
base_y = []
with open('random_insertion_results.txt', 'r') as f:
    for line in f:
        overhead = float(line.split(',')[0].split('=')[1].strip())
        acc = float(line.split(',')[1].split('=')[1].strip())
        base_x.append(overhead)
        base_y.append(acc)

plt.plot(x, y, 'bs', base_x, base_y, 'r--')
plt.xlabel('Overhead')
plt.ylabel('Accuracy')
plt.title('Performance on Test Data')
plt.savefig('evo_performance.png')
