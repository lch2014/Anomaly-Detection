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
    if data_name == "ids2018":
        for row in md8.FEATURE_LIST[0:1]
        for row in md8.FEATURE_LIST[2:]:
            df[row] = df[row].astype(float)
    return df

#标签编码
def label_encoding(df, labels, option, data_name):
    if data_name == "ids2018":
        #小类编码
        if option == 1:
            codes = [ i for i in range(len(labels))]

        #大类编码
        elif option == 2:
            codes = [0, 1, 2, 2, 2, 2, 3, 4, 4, 5, 4, 4, 6, 6]

     #二分编码
        elif option == 3:
            codes = []
            if data_name == "ids2018"
                for label in labels:
                    if label == "Benign":
                        codes.append(0)
                    else:
                        codes.append(1)

        df['Label'].replace(labels, codes, inplace=True)
    return df