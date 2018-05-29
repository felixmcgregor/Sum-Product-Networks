from SumNode import SumNode
from ProductNode import ProductNode
from Node import LeafNode
from SPN import SPN

import numpy as np
import math
from sklearn.metrics import mutual_info_score
from sklearn.mixture import GaussianMixture
import random
import pandas as pd


def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi * 2


def learnSPN(data_matrix):

    spn = SPN()

    # split for sum node on the first one
    root_node = SumNode(0, 0)
    spn.add_node(root_node)
    sub_split(data_matrix, spn, 0, False)

    return spn


def sub_split(data_matrix, spn, parent_id, split_independent=False):

    # return when there is only one instance
    if len(data_matrix) <= 1:
        print("terminate\n")
        # print(data_matrix)

        # make leazf node
        return

    elif split_independent is False:
        # split into similar instances
        cluster(data_matrix, spn, parent_id)

    elif split_independent is True:
        # split independent variables
        split_variables(data_matrix, spn, parent_id)


def cluster(data_matrix, spn, parent_id):

    print("Clustering...\n")
    n_clusters = 2
    # Find clusters
    gmm = GaussianMixture(n_components=n_clusters)
    gmm.fit(data_matrix)
    clusters = gmm.predict(data_matrix)
    for i in range(n_clusters):
        # add product node
        prod_node = ProductNode(parent_id + 1 + i, parent_id)
        spn.add_node(prod_node, len(data_matrix[clusters == i]))

        # split on recursively submatrix
        sub_matrix = data_matrix[clusters == i]
        print("Split clustered \n", sub_matrix)
        sub_split(sub_matrix, spn, parent_id + 1 + i, True)
        print("########################################################################")


def split_variables(data_matrix, spn, parent_id):
    print("\nFinding independencies...\n")
    flagged = []
    variables = data_matrix.T
    # shuffle
    variables = variables.sample(frac=1)

    # iterate through all the varibales randomly
    for index, row in variables.iterrows():

        if index not in flagged:
            flagged.append(index)

            # compare with other variables
            for index1, row1 in variables.iterrows():

                # Check independence between selected variable and other variables
                mutual_info = calc_MI(row, row1, 2)
                if mutual_info > 0.05 and index1 not in flagged:
                    flagged.append(index1)

                    # check independence between similar variable and other variables
                    for index2, row2 in variables.iterrows():
                        mutual_info2 = calc_MI(row1, row2, 2)
                        if mutual_info2 > 0.05 and index2 not in flagged:
                            flagged.append(index2)

            print("\nDependent variables", flagged, "Starting subsplit\n")

            arr = np.array(data_matrix)

            if len(flagged) == 1 or (arr == arr[0]).all():
                # make leaf if there is only one variable

                if (arr == arr[0]).all():
                    print("instances are the same")
                print("Making leaf for ", flagged)
                X = LeafNode(parent_id, np.array([1]), parent_id, False)
                X_ = LeafNode(parent_id, np.array([1]), parent_id, True)
                spn.add_node(X)
                spn.add_node(X_)

            else:
                sum_node = SumNode(parent_id + 1, parent_id)
                spn.add_node(sum_node)
                sub_matrix = variables.ix[flagged]
                print("making sum node")
                sub_split(sub_matrix.T, spn, parent_id + 1, False)

        else:
            print("yolo")
