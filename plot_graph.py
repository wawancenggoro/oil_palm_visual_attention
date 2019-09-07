import matplotlib.pyplot as plt
import pandas
import csv
import os
import re

pattern = re.compile('.+#.+\.csv')
for filename in os.listdir('results_csv/plot_target'):
    if pattern.match(filename):
        csv_name = re.search('#.+', filename).group(0)[1:-4]
        x_label = 'epochs'
        train_label_acc = 'train_acc'
        val_label_acc = 'val_acc'

        train_label_loss = 'train_loss'
        val_label_loss = 'val_loss'

        df = pandas.read_csv(f"results_csv/plot_target/{filename}")

        df[train_label_acc] = pandas.to_numeric(df[train_label_acc])
        df[val_label_acc] = pandas.to_numeric(df[val_label_acc])
        df[train_label_loss] = pandas.to_numeric(df[train_label_loss])
        df[val_label_loss] = pandas.to_numeric(df[val_label_loss])

        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle(f"{csv_name}")

        ax1.plot(df[df.columns[0]], df[train_label_acc])
        ax1.plot(df[df.columns[0]], df[val_label_acc])
        ax1.set(ylabel="Accuracy")

        ax2.plot(df[df.columns[0]], df[train_label_loss])
        ax2.plot(df[df.columns[0]], df[val_label_loss])
        ax2.set(ylabel="Loss")
        if not os.path.exists(f'results_csv/plot_images/{csv_name}.png'):
            fig.savefig(f'results_csv/plot_images/{csv_name}.png')

        plt.close()
        # fig.savefig(f'results_csv/plot_images/{csv_name}.png', bbox_inches="tight")

        # plt.ylim(df[y_label].min() - 0.05 if df[y_label].min() > 0.05 else df[y_label].min(),
        #          df[y_label].max() + 0.05 if df[y_label].max() < 0.95 else 1)
