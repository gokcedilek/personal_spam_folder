import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def label_metric(ax, bars):
    # display metric value (y value) for each *bar* in *bars*
    for bar in bars:
        height = round(bar.get_height(), 3)
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=15)


def plot_metric(train_df, test_df, metric):
    # for each metric, create a (grouped) bar graph of trial # (x axis) vs training & test value of the metric (y axis)

    # width of training / test bars
    bar_width = 0.15

    # number of group labels to add = number of rows in the dataframe
    indices = list(train_df.index.values)
    # x positions of group labels (middle point of training & test bars)
    x_labels = np.arange(len(indices))

    # y positions for training bars
    train_y = list(train_df[metric])
    # y positions for test bars
    test_y = list(test_df[metric])

    fig, ax = plt.subplots()
    # define training bars
    train_bars = ax.bar(x_labels - bar_width/2, train_y,
                        bar_width, label='Training')
    # define test bars
    test_bars = ax.bar(x_labels + bar_width/2, test_y, bar_width, label='Test')

    # add metric title
    ax.set_ylabel(metric)

    # add group labels
    ax.set_xticks(x_labels)
    ax.set_xticklabels(x_labels)

    # add legend (to outside of the figure)
    ax.legend(bbox_to_anchor=(1, 1))

    # display metric values with the bars
    label_metric(ax, train_bars)
    label_metric(ax, test_bars)

    # save the metric plot
    fig.tight_layout()
    plt.savefig(f'{metric}.png')


def plot_stats(training_csv, test_csv):
    if (os.path.isfile(training_csv) and os.path.isfile(test_csv)):
        train_df = pd.read_csv(training_csv)
        test_df = pd.read_csv(test_csv)
        metrics = list(train_df.columns)
        # create one train-test plot for each metric
        for metric in metrics:
            plot_metric(train_df, test_df, metric)
