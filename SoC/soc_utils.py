import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import ce_loss

def normalize(x):
    x_sum = torch.sum(x)
    x = x / x_sum
    return x.detach()

class Get_Scalar:
    def __init__(self, value):
        self.value = value
        
    def get_value(self, iter):
        return self.value
    
    def __call__(self, iter):
        return self.value

def calcDis(dataSet, centroids, k):
    clalist=[]
    
    for i in range(np.size(dataSet,0)):
        temp = []
        for c in range(k):
            dist = np.linalg.norm(dataSet[i] - centroids[c])
            temp.append(dist)
        clalist.append(temp)
        
    return clalist

def classify(dataSet, centroids, k, reverse):
    clalist = calcDis(dataSet, centroids, k)
    if reverse:
        minDistIndices = np.argmin(clalist, axis=1)  
    else:  
        minDistIndices = np.argmax(clalist, axis=1)    
    cluster = [[centroids[i]] for i in range(k)]
    for x in range(len(minDistIndices)):
        if not x in cluster[minDistIndices[x]] and not x in centroids:
            cluster[minDistIndices[x]].append(x)
    newCentroids = []
    subgraph = []
    for x in cluster:
        temp = [[] for i in range(len(x))]
        for i in range(len(x)):
            for y in x:
                temp[i].append(dataSet[x[i]][y])
        subgraph.append(temp)
    for i in range(k):
        min_value = []
        for row in subgraph[i]:
            min_value.append(np.sum(row))
        if reverse:
            newCentroids.append(cluster[i][np.argmin(min_value)])  
        else:  
            newCentroids.append(cluster[i][np.argmax(min_value)])   
    changed = set(newCentroids) == set(centroids)
    return changed, newCentroids

def kmeans(dataSet, k, centroids):
    min_value = []
    if centroids is None:
        centroids = dataSet[np.random.choice(dataSet.shape[0], k, replace=False)]
    changed, newCentroids = classify(dataSet, centroids, k, True)
    n = 0
    while not changed and n < 2000:
        changed, newCentroids = classify(dataSet, newCentroids, k, True)
        n += 1
    clalist = calcDis(dataSet, newCentroids, k)
    minDistIndices = np.argmax(clalist, axis=1)  
    cluster = [[centroids[i]] for i in range(k)]
    for x in range(len(minDistIndices)):
        if not x in cluster[minDistIndices[x]] and not x in centroids:
            cluster[minDistIndices[x]].append(x)
    cluster = [cl if cl else [None] for cl in cluster]        
    dic = {}
    for i, j in enumerate(cluster): 
        for x in j:
            dic[x] = i
    for i in range(len(dataSet)):
        if i not in dic:
            dic[i] = None  # If index is missing, assign to None        
    return dic, cluster, newCentroids

def kmeans_reverse(dataSet, k, centroids):
    min_value = []
    changed, newCentroids = classify(dataSet, centroids, k, False)
    n = 0
    while not changed and n < 2000:
        changed, newCentroids = classify(dataSet, newCentroids, k, False)
        n += 1
    clalist = calcDis(dataSet, newCentroids, k)
    minDistIndices = np.argmin(clalist, axis=1)  
    cluster = [[centroids[i]] for i in range(k)]
    for x in range(len(minDistIndices)):
        if not x in cluster[minDistIndices[x]] and not x in centroids:
            cluster[minDistIndices[x]].append(x)
    dic = {}
    for i, j in enumerate(cluster): 
        for x in j:
            dic[x] = i
    return dic, cluster, newCentroids

def consistency_loss_soc(logits_w, label_dics, clusters, alpha, num_classes):
    logits_w = logits_w.detach()
 
    pseudo_label = torch.softmax(logits_w, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)

    num_cluster = round(num_classes/alpha)
    filter_value = float(0)
    p_temp = pseudo_label
    for idx, p in enumerate(pseudo_label):
        max_probs_p, max_idx_p = torch.max(p, dim=-1)
        conf_idx = round((max_probs_p.cpu().item() ) * num_cluster)
        # Ensure conf_idx is valid
        if conf_idx >= len(clusters):
            print(f"Invalid conf_idx: {conf_idx} exceeds cluster length: {len(clusters)}")
            continue

        # Ensure label_dic exists for conf_idx
        if conf_idx not in label_dics:
            print(f"conf_idx {conf_idx} not found in label_dics")
            continue

        # Ensure max_idx value is valid for label_dic
        if max_idx[idx].cpu().item() >= len(label_dics[conf_idx]):
            print(f"Invalid max_idx: {max_idx[idx].cpu().item()} for label_dic[{conf_idx}]")
            continue
        indices_to_remain = clusters[conf_idx][label_dics[conf_idx][max_idx[idx].cpu().item()]]
        indices_to_remove = list(set([i for i in range(num_classes)]) - set(indices_to_remain))
        p[indices_to_remove] = filter_value
        p = normalize(p)
        p_temp[idx] = p 
    pseudo_label = p_temp 
    loss_super = ce_loss(logits_w, pseudo_label, use_hard_labels = False, reduction='none') 
    return loss_super.mean()    

