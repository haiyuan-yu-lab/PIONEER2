import numpy as np
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, precision_recall_curve, accuracy_score

def compute_roc(labels, preds):
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_aupr(labels, preds):
    p, r, _ = precision_recall_curve(labels.flatten(), preds.flatten())
    aupr = auc(r, p)
    return aupr

def compute_mcc(labels, preds):
    labels = labels.astype(np.float64)
    preds = preds.astype(np.float64)
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc

def acc_score(labels, preds):
    acc = accuracy_score(labels.flatten(), preds.flatten())
    return acc

def compute_performance_max(labels, preds, beta=1.0):
    predictions_max = None
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    for t in range(1, 1000):
        threshold = t / 1000.0
        predictions = (preds > threshold).astype(np.int32)
        
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        
        if tp + fp > 0:
            p = tp / (1.0 * (tp + fp))
        else:
            p = 0.0
        if tp + fn > 0:
            r = tp / (1.0 * (tp + fn))
        else:
            r = 0.0
        if p + r > 0:
            f = (1+beta*beta) * p * r / (beta*beta*p + r)
            if f > f_max:
                f_max = f
                p_max = p
                r_max = r
                t_max = threshold
                predictions_max = predictions
        else:
            f = 0.0
    return f_max, p_max, r_max, t_max, predictions_max

def compute_performance(labels, preds, threshold, beta=1.0):
    predictions = (preds > threshold).astype(np.int32)
    tp = np.sum(labels * predictions)
    fp = np.sum(predictions) - tp
    fn = np.sum(labels) - tp

    if tp + fp > 0:
        p = tp / (1.0 * (tp + fp))
    else:
        p = 0.0
    if tp + fn > 0:
        r = tp / (1.0 * (tp + fn))
    else:
        r = 0.0
    if p + r > 0:
        f = (1+beta*beta) * p * r / (beta*beta*p + r)
    else:
        f = 0.0
    return f, p, r, predictions
