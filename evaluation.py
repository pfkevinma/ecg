# Confusion Matrix
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import os

ground_truth_filepath = os.getcwd()+"/examples/cinc17/data/sample2017/ansfile.txt" # start from root
prediction_filepath = os.getcwd()+"/answer.txt"


y_predict = []
y_true = []

data = {}

with open(ground_truth_filepath, "r") as f:
	for line in f:
 		y_true.append(line.split(",")[1].strip())
 	data["y_true"] = y_true

with open(prediction_filepath, "r") as f:
	for line in f:
		y_predict.append(line.split(",")[0].strip())
	data["y_predict"] = y_predict


print(len(data["y_true"]), len(data["y_predict"]))


df = pd.DataFrame(data, columns=['y_true','y_predict'])
confusion_matrix = pd.crosstab(df['y_true'], df['y_predict'], rownames=['true'], colnames=['predict'])
print (confusion_matrix)

sn.heatmap(confusion_matrix, annot=True)
plt.show()