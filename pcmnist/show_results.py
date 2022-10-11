# -*- coding: utf-8 -*-
import numpy as np
import seaborn as sns
import json
import pandas as pd
import os
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", 
                        type=str, 
                        required=True,
                        default=None,
                        help="""a string describing results file path""")

def main():
    option = parser.parse_args()
    train_num = 10000
    val_num = 5000
    result_file_path = option.file
    result_file = os.path.join(result_file_path, 'metrics.json')
    with open(result_file) as f:
        results = json.load(f)

    epochs = []
    train_acc = []
    test_acc = []
    val_acc = []
    val_tev_acc = []
    for epoch_num in results['Main'].keys():
        epochs.append(int(epoch_num))
        train_acc.append(results['Main'][epoch_num]['Train']['accuracy'])
        test_acc.append(results['Main'][epoch_num]['Test']['accuracy'])
        val_acc.append(results['Main'][epoch_num]['Val']['accuracy'])
        val_tev_acc.append(results['Main'][epoch_num]['Val_TEV']['accuracy'])

    df = pd.DataFrame({"Train": train_acc,
                    "Val": val_acc,
                    "Test": test_acc,
                    "Val_TEV": val_tev_acc,
                    "epoch": epochs})
    sns.lineplot(x='epoch', y='Train', data=df)
    sns.lineplot(x='epoch', y='Val', data=df)
    sns.lineplot(x='epoch', y='Test', data=df)
    sns.lineplot(x='epoch', y='Val_TEV', data=df)
    plt.legend(labels=['Train', 'Val', 'Test', "Val_TEV"])
    figure_file = os.path.join(result_file_path, 'output.png')
    plt.savefig(figure_file)
    print("Finished.")

if __name__ == '__main__':
    main()