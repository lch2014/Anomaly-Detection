import pandas as pd
import sys
import glob
import csv_utils as cu
import meta_data_2018 as md8

#根据标签分割数据集
def split_datasets_by_label(df, data_name, dst_path):
    if data_name == "ids2018":
        for label in md8.LABEL_LIST:
            print("***** Handling " + str(label) + " *******")
            td = cu.split_dataset_by_label(df, [label])
            cu.write_to_csv(td, dst_path + str(label) + ".csv")
    else:
        pass

#根据标签分割的数据集,构建子集
def make_subsets_by_ratio(file_path, ratio, dst_path):
    all_data = []
    for f in glob.glob(file_path + "*.csv"):
        df = pd.read_csv(f, low_memory=False)
        td = cu.split_dataset_by_ratio(df, ratio)
        all_data.append(td)
    data = pd.concat(all_data, axis=0, ignore_index=True)
    return data

#转换数据类型
def convert_datatype(df, data_name):
    pass