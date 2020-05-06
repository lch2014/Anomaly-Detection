import pandas as pd
import sys
from csv_utils import *
import meta_data_2018 as md8

#根据标签分割数据集
def split_datasets_by_label(df, data_name, target_path):
    if data_name == "ids2018":
        for label in md8.LABEL_LIST:
            print("***** Handling " + str(label) + " *******")
            td = split_dataset_by_label(df, [label])
            write_to_csv(td, target_path + str(label) + ".csv")
    else:
        pass

#根据标签分割的数据集,构建子集
def make_subsets_by_ratio(file_path, ratio):
    pass