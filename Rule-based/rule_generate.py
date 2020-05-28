from sklearn import tree
from sklearn.tree import _tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

def rule_generate(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined" for i in tree_.feature ]
    print(feature_name)

    def recurse(node, rule):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            left_str = "if {} <= {}".format(name, threshold)
            right_str = "if {} > {}".format(name, threshold)
            recurse(tree_.children_left[node], rule + " " + left_str)
            recurse(tree_.children_right[node], rule + " " + right_str)
        else:
            print(rule + ", then {}".format(tree_.value[node]))

    recurse(0, "")

if __name__ == "__main__":
    pass