import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score

def get_avg_f1(y_true, y_pred):
    labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    # we take the average of best f1 match scores as defined in Yang et al.
    avg_f1_score = 0
    for l in labels:
        #fit pred to true
        c1 = np.where(y_true == l, 1, 0)
        g1_scores = [f1_score(c1, np.where(y_pred == c, 1, 0)) for c in labels]
        g1 = np.max(g1_scores) / len(labels)
        
        if l in pred_labels:
            g2_scores = [f1_score(np.where(y_true == c, 1, 0), 
                                  np.where(y_pred == l, 1, 0)) 
                                  for c in pred_labels]
            g2 = np.max(g2_scores) / len(pred_labels)
            avg_f1_score += g2
        
        avg_f1_score += g1
    
    avg_f1_score /= 2
    
    return avg_f1_score

def topic_stats(embeddings, y_true, num_classes, geo_vec=False):
    """ Computes topic cluster statistics for document vectors
    Param:
        embeddings: (geo_vec), (D, B, C) tensor, where D is num documents, 
                                BxD final embedding.
                    (D, C) tensor, where D is num documents, C embedding.
        y_true: (D, 1), correct topic per document
        num_classes: total number of classes in the dataset 
        geo_vec: check the simple topic activation
    Returns:
        f1_s: weighted f1 score
        ari: adjusted rand score
        ami: adjusted mutual information score
        nmi: normalized mutual information score
        topic_s: maxed topic activations
    """
    if geo_vec:
        topic_act = np.argmax(np.sum(embeddings, axis=2), axis=1)
        embeddings = np.reshape(c, (np.shape(c)[0], -1))

    kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(embeddings)
    
    f1_s, ari, ami, nmi, topic_s = 0, 0, 0, 0, 0
    def score(y_true, y_pred, name):
        n = ['avg_f1', 'adj_ri', 'adj_mi', 'norm_mi', 'topic_act']
        s = [get_avg_f1(y_true, y_pred), adjusted_rand_score(y_true, y_pred), 
              adjusted_mutual_info_score(y_true, y_pred),
              normalized_mutual_info_score(y_true, y_pred),float('NaN')]
        if geo_vec:
            s[-1] = (accuracy_score(y_true, topic_act))
        print ('      %10s  %10s  %10s  %10s  %10s \n %3s %8.3f  %10.3f  %10.3f  %10.3f  %10.3f\n' % (
               n[0],n[1],n[2],n[3],n[4],name,s[0],s[1],s[2],s[3], s[4]))
        return s

    cl_methods = [kmeans.labels_]
    cl_names = ['kmeans']
    
    y_pred = []
    for y_pred_, name in zip(cl_methods, cl_names):
        f1_s, ari, ami, nmi, topic_s = score(y_true, y_pred_, name)
        
    return f1_s, ari, ami, nmi, topic_s

a = np.array([[[0, 5],[2, 2]]])
b = np.array([[[0, 1],[2, 2]]])
# geo_vec model example
c = np.vstack((a, b))
# regular doc embedding example
#c = np.squeeze(a)
cl = np.array([0, 1])
topic_stats(c, cl, 2, True)


