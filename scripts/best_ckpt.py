import sys
import numpy as np

input_file = sys[1]

records = open(input_file).readlines()

results = []
for re in records:
    values = re.strip().split("")[1].split(', ')
    values = list(map(lambda x: float(x.split('=')[1]), values))
    results.append(values)

results = np.asarray(results)
mean = np.max(results, axis=0)
print(mean)