#P1B: : Implement and demonstrate the FIND-S algorithm for finding the most specific hypothesis based on a given set of training data samples. Read the training data from a .CSV file.
#Dataset: P1B.csv

import csv

print('Saail Chavan 016')

# Variable Declaration
num_attributes = 6
data_set = []

# Reading Data set from .csv file
with open('/content/sample_data/P1B.csv', "r") as csvfile:
  reader = csv.reader(csvfile)
  for row in reader:
    data_set.append(row)

print("Data Set is :")
print(data_set)

hypothesis = ['0'] * num_attributes
print('Null hypothesis')
print(hypothesis)

for j in range(0, num_attributes):
  hypothesis[j] = data_set[1][j]
  for i in range(1, len(data_set)):
    if data_set[i][num_attributes] == 'Yes':
            for j in range(1, num_attributes):
                if data_set[i][j] != hypothesis[j]:
                    hypothesis[j] = '?'
                else:
                    hypothesis[j] = data_set[i][j]
                    print("For Training instance no:{0} the hypothesis is ".format(i), hypothesis)

print("\n The Maximally Specific Hypothesis for a given training examples: \n")
print(hypothesis)
print('Saail Chavan 016')
