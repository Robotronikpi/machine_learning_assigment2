""" This file is created as a suggested solution template for question 2.2 in DD2434 - Assignment 2.

    We encourage you to keep the function templates as is.
    However this is not a "must" and you can code however you like.
    You can write helper functions however you want.

    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py
    file), and modify them as needed. In addition to the sample files given to you, it is very important for you to
    test your algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format. Let us know if you face any problems.

    Also, we are aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). We wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).

    For this assignment, we gave you three different trees (q_2_2_small_tree, q_2_2_medium_tree, q_2_2_large_tree).
    Each tree has 5 samples (whose inner nodes are masked with np.nan values).
    We want you to calculate the likelihoods of each given sample and report it.

    Note: The alphabet "K" is K={0,1,2,3,4}.
"""
import time

def child_u(tree,u):
    b = np.where(tree == u)
    return b[0]


def is_leaf(beta,v):
    if(str(beta[v])!='nan'):
        return True
    return False

def condi(theta,child,j,i):
    return theta[child][i][j]

def S(K,tree,beta,theta,u,i):
    global dico

    child = child_u(tree,u)
    v=0
    w=0
    if (str([child[0],i]) in dico):
        v =  dico[str([child[0],i])]
    else :
        if(is_leaf(beta,child[0])):
            v=condi(theta,child[0],int(beta[child[0]]),i)
        else:
            for j in range(K):
                v+=S(K,tree,beta,theta,child[0],j)*condi(theta,child[0],j,i)
        dico[str([child[0],i])]=v

    if (str([child[1],i]) in dico):
        w = dico[str([child[1],i])]
    else :
        if(is_leaf(beta,child[1])):
            w=condi(theta,child[1],int(beta[child[1]]),i)
        else:
            for jj in range(K):
                w+=S(K,tree,beta,theta,child[1],jj)*condi(theta,child[1],jj,i)
        dico[str([child[1],i])]=w
    return v*w





import numpy as np
from Tree import Tree
from Tree import Node


def calculate_likelihood(tree_topology, theta, beta,K):
    """
    This function calculates the likelihood of a sample of leaves.
    :param: tree_topology: A tree topology. Type: numpy array. Dimensions: (num_nodes, )
    :param: theta: CPD of the tree. Type: numpy array. Dimensions: (num_nodes, K, K)
    :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
                Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
    :return: likelihood: The likelihood of beta. Type: float.

    This is a suggested template. You don't have to use it.

    """
    likelihood = 0
    for j in range(K):
        likelihood += S(K,tree_topology,beta,theta,0,j)*theta[0][j]
    # TODO Add your code here
    print("tree_topology", beta)
    # Start: Example Code Segment. Delete this segment completely before you implement the algorithm.
    print("Calculating the likelihood...")
    # likelihood = np.random.rand()
    # End: Example Code Segment

    return likelihood

dico = {}
def main():
    global dico
    print("Hello World!")
    print("This file is the solution template for question 2.2.")

    print("\n1. Load tree data from file and print it\n")

    filename = "data/q2_2/q2_2_large_tree.pkl"  # "data/q2_2/q2_2_medium_tree.pkl", "data/q2_2/q2_2_large_tree.pkl"
    t = Tree()
    t.load_tree(filename)
    t.print()
    print("K of the tree: ", t.k, "\talphabet: ", np.arange(t.k))

    print("\n2. Calculate likelihood of each FILTERED sample\n")
    # These filtered samples already available in the tree object.
    # Alternatively, if you want, you can load them from corresponding .txt or .npy files
    time_tot1 = time.time()
    for sample_idx in range(t.num_samples):
        tps1 = time.time()
        dico = {}
        beta = t.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        sample_likelihood = calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta,t.k)
        tps2 = time.time()
        print("\tLikelihood: ", sample_likelihood)
        print("time :", tps2 - tps1)
    time_tot2 = time.time()
    print("time_tot",(time_tot2-time_tot1)/(sample_idx+1))
    beta = np.empty(5)
    beta[:] = np.NaN
    sum = 0
    # for i in range(5):
    #     for j in range(5):
    #         for k in range(5):
    #             dico = {}
    #             beta[2] = i
    #             beta[3] =j
    #             beta[4] = k
    #             sum+=calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta,t.k)
    # print(sum)
if __name__ == "__main__":
    main()
