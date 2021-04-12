import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def plot():
	ground_truth_filepath = os.getcwd()+"/data/sample2017/answers.txt"
	#prediction_filepath = os.getcwd()+"/entry/entry/answers.txt"
	prediction_filepath = "/Users/peifengma/Documents(Mac)/ecg/saved/cinc17/cinc17_results/answers.txt"
	
	labels = ["N", "A", "O", "~"]
	data = {}
	
	with open(ground_truth_filepath, "r") as f:
		y_true = [line.split(",")[1].strip() for line in f]
		y_true_code =[labels.index(ele) for ele in y_true]
		data["y_true"] = y_true_code
	with open(prediction_filepath, "r") as f:
		y_predict = [line.split(",")[1].strip() for line in f]
		y_predict_code =[labels.index(ele) for ele in y_predict]
		data["y_predict"] = y_predict_code

	
	df = pd.DataFrame(data, columns=['y_true','y_predict'])
	confusion_matrix = pd.crosstab(df['y_true'], df['y_predict'], rownames=['true'], colnames=['predict'])
	print(confusion_matrix)
	
	sn.heatmap(confusion_matrix, annot=True)
	plt.show()



def main():
	cwd = os.getcwd()
	ground_truth_filepath = cwd+"/data/sample2017/answers.txt"
	#prediction_filepath = cwd+"/entry/entry/answers.txt"
	prediction_filepath = "/Users/peifengma/Documents(Mac)/ecg/saved/cinc17/cinc17_results/answers.txt"

	true_label ={}
	with open(ground_truth_filepath, "r") as f:
		for line in f:
			line = line.split(",")
			true_label[line[0]]= line[1].strip()

	prediction = {}
	with open(prediction_filepath, "r") as f:
		for line in f:
			line = line.split(",")
			prediction[line[0]]= line[1].strip()

	labels = ["N", "A", "O", "~"]

	matrix = []

	for label in labels:
		lst = [0, 0, 0, 0]
		for key, value in true_label.items():
			if value == label:
				if key in prediction:
					lst[labels.index(prediction[key])]+=1
		matrix.append(lst)

	matrix_array = np.asarray(matrix)
	print(len(prediction))
	print(matrix_array)

if __name__ == "__main__":
	main()
	plot()


