import os
import pandas as pd

def selectFile(file):
    filters = [dataset]
    for f in filters:
        if f not in file:
            return False
    return True

dataset = "fmnist"
type = "5_GradientNoise"
path = "../save_opt/{}/Final/".format(type)
s = os.listdir(path)
files = list(filter(selectFile, s))
print(files)
dict = {'Epoch': range(10,101,10)}

for file in files:
    f = open(path+file, "r")
    res = f.readline()
    res = res.split(', ')
    print(len(res))
    dict[file[:file.find('_')]] = res
dataframe = pd.DataFrame(dict)
print(dataframe)
dataframe.to_csv("../save_opt/{}/result_{}.csv".format(type, dataset), index=False, sep=' ')

