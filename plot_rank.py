import matplotlib.pyplot as plt
import pandas
import csv
import os
import re
from pprint import pprint

training_group = {}
val_group = {}

epoch_count = range(0, 50)

pattern = re.compile('.+#.+\.csv')
for filename in os.listdir('results_csv/plot_target'):
    if pattern.match(filename):
        csv_name = re.search('#.+', filename).group(0)[1:-4].split('_')
        train_label_acc = 'train_acc'
        val_label_acc = 'val_acc'

        df = pandas.read_csv(f"results_csv/plot_target/{filename}")

        training_group[f"{csv_name[0]}_{csv_name[2]}_{csv_name[3][0]}"] = pandas.to_numeric(df[train_label_acc])
        val_group[f"{csv_name[0]}_{csv_name[2]}_{csv_name[3][0]}"] = pandas.to_numeric(df[val_label_acc])

training_highest = []
val_highest = []

for i, group_name in enumerate(training_group):
        group_highest_value = 0
        for group_value in training_group[group_name]:
                if group_highest_value < group_value:
                        group_highest_value = group_value
        training_highest.append({'name': group_name, 'value': group_highest_value})


for i, group_name in enumerate(val_group):
        group_highest_value = 0
        for group_value in val_group[group_name]:
                if group_highest_value < group_value:
                        group_highest_value = group_value
        val_highest.append({'name': group_name, 'value': group_highest_value})

training_highest = sorted(training_highest, key=lambda x: x['value'], reverse=True)
val_highest = sorted(val_highest, key=lambda x: x['value'], reverse=True)

print("Training Highest:")
pprint(training_highest)
print("\n\n")

print("Validation Highest:")
pprint(val_highest)
