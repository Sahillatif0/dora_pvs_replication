"""
Challenge Evaluation Suite for DoRA-PVS.
Implements:
1. Voxel-wise AUPRC (Area Under the Precision-Recall Curve)
2. clDice (Topology-preserving Centreline Dice)
3. Lesion-wise DSC (Instance overlap using the one-inside criterion)
"""

import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from skimage.morphology import skeletonize
from scipy.ndimage import label

def compute_auprc(y_true_binary, y_pred_prob, roi_mask=None):
    """
    Computes voxel-wise Area Under the Precision-Recall Curve (AUPRC).
    """
    if roi_mask is not None:
        y_true_binary = y_true_binary[roi_mask > 0]
        y_pred_prob = y_pred_prob[roi_mask > 0]
    else:
        y_true_binary = y_true_binary.flatten()
        y_pred_prob = y_pred_prob.flatten()

    if np.sum(y_true_binary) == 0:
        return 0.0

    precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_prob)
    return float(auc(recall, precision))

def compute_cldice(y_true_binary, y_pred_binary):
    """
    Computes Centreline Dice (clDice):
    clDice = 2 * (Tprec * Tsens) / (Tprec + Tsens)
    where Tprec = |Vp ∩ Sl| / |Sl|, Tsens = |Vl ∩ Sp| / |Sp|
    """
    y_true_bool = y_true_binary.astype(bool)
    y_pred_bool = y_pred_binary.astype(bool)

    if not np.any(y_true_bool) and not np.any(y_pred_bool):
        return 1.0
    if not np.any(y_true_bool) or not np.any(y_pred_bool):
        return 0.0

    # Extract 3D skeletons / centrelines
    s_true = skeletonize(y_true_bool) > 0
    s_pred = skeletonize(y_pred_bool) > 0

    t_prec = np.sum(y_pred_bool & s_true) / (np.sum(s_true) + 1e-7)
    t_sens = np.sum(y_true_bool & s_pred) / (np.sum(s_pred) + 1e-7)

    if (t_prec + t_sens) == 0:
        return 0.0

    cldice = 2.0 * (t_prec * t_sens) / (t_prec + t_sens)
    return float(cldice)

def compute_lesion_wise_dsc(y_true_binary, y_pred_binary):
    """
    Computes Lesion-wise DSC using the standard one-inside hit criterion.
    Counts individual connected PVS segments detected by predicted clusters.
    """
    true_labels, num_true = label(y_true_binary.astype(int))
    pred_labels, num_pred = label(y_pred_binary.astype(int))

    if num_true == 0 and num_pred == 0:
        return 1.0
    if num_true == 0 or num_pred == 0:
        return 0.0

    tp = 0
    # For each true lesion instance, check if any predicted voxel intersects it
    for i in range(1, num_true + 1):
        instance_mask = (true_labels == i)
        if np.any(pred_labels[instance_mask] > 0):
            tp += 1

    fp = 0
    # For each predicted lesion, check if it fails to hit any ground truth voxel
    for j in range(1, num_pred + 1):
        pred_mask = (pred_labels == j)
        if not np.any(true_labels[pred_mask] > 0):
            fp += 1

    fn = num_true - tp
    dsc = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)
    return float(dsc)
