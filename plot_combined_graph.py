import matplotlib.pyplot as plt
import pandas
import csv
import os
import re
import sys


training_group = {}
val_group = {}

colors_group = []
for color in ['#1abc9c', '#e056fd', '#0652DD', '#27ae60', '#5f27cd', '#ff7979', '#3498db', '#9b59b6', '#34495e', '#f1c40f', '#e67e22', '#e74c3c', '#B53471', '#6ab04c', '#006266', '#b33939']:
        colors_group.append(color)
        colors_group.append(color)

opt_title = sys.argv[2]
pattern = re.compile(f'.+#.+({opt_title}).+\.csv')
for filename in sorted(os.listdir('results_csv/plot_target')):
    if pattern.match(filename):
        csv_name = re.search('#.+', filename).group(0)[1:-4].split('_')
        train_label_acc = 'train_acc'
        val_label_acc = 'val_acc'

        df = pandas.read_csv(f"results_csv/plot_target/{filename}")

        if (df.shape[0] > 45):
                training_group[f"{csv_name[0]}_{csv_name[2]}_{csv_name[3][0]}"] = pandas.to_numeric(df[train_label_acc])
                val_group[f"{csv_name[0]}_{csv_name[2]}_{csv_name[3][0]}"] = pandas.to_numeric(df[val_label_acc])

targetted_group = training_group if sys.argv[1] == 'Training' else val_group


fig = plt.figure()
ax = plt.subplot(111)
plt.title(f'{sys.argv[1]} {opt_title}')

for i, group_name in enumerate(targetted_group):
        plot_line_style = 'dashed' if i % 2 == 0 else 'solid'
        ax.plot(range(0, len(targetted_group[group_name])), targetted_group[group_name], linestyle=plot_line_style, color=colors_group[i], label=group_name)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

# plt.legend(loc='upper center', bbox_to_anchor=(-0.03, 1.15), ncol=1)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()
