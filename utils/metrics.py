# https://medium.datadriveninvestor.com/a-survey-of-evaluation-metrics-for-multilabel-classification-bb16e8cd41cd

import numpy as np
from sklearn.metrics import precision_score, accuracy_score


def emr(y_true, y_pred):
    n = len(y_true)
    row_indicators = np.all(y_true == y_pred, axis = 1) # axis = 1 will check for equality along rows.
    exact_match_count = np.sum(row_indicators)
    return exact_match_count/n


def hamming_loss(y_true, y_pred):
    """
	XOR TT for reference - 
	
	A  B   Output
	
	0  0    0
	0  1    1
	1  0    1 
	1  1    0
	"""
    hl_num = np.sum(np.logical_xor(y_true, y_pred))
    hl_den = np.prod(y_true.shape)
    
    return hl_num/hl_den


def example_based_accuracy(y_true, y_pred):
    
    # compute true positives using the logical AND operator
    numerator = np.sum(np.logical_and(y_true, y_pred), axis = 1)

    # compute true_positive + false negatives + false positive using the logical OR operator
    denominator = np.sum(np.logical_or(y_true, y_pred), axis = 1)
    instance_accuracy = numerator/denominator

    avg_accuracy = np.mean(instance_accuracy)
   
    # print(accuracy_score(y_true, y_pred))
    return avg_accuracy


def example_based_precision(y_true, y_pred):
    """
    precision = TP/ (TP + FP)
    """
    avg_precision = precision_score(y_true, y_pred, average='samples', zero_division=1)
    
    return avg_precision


def label_based_macro_accuracy(y_true, y_pred):
	
    # axis = 0 computes true positives along columns i.e labels
    l_acc_num = np.sum(np.logical_and(y_true, y_pred), axis = 0)

    # axis = 0 computes true postive + false positive + false negatives along columns i.e labels
    l_acc_den = np.sum(np.logical_or(y_true, y_pred), axis = 0)

    # compute mean accuracy across labels. 
    return np.mean(l_acc_num/l_acc_den)




def label_based_macro_precision(y_true, y_pred):
	
	# axis = 0 computes true positive along columns i.e labels
	l_prec_num = np.sum(np.logical_and(y_true, y_pred), axis = 0)

	# axis = computes true_positive + false positive along columns i.e labels
	l_prec_den = np.sum(y_pred, axis = 0)

	# compute precision per class/label
	l_prec_per_class = l_prec_num/l_prec_den

	# macro precision = average of precsion across labels. 
	l_prec = np.mean(l_prec_per_class)
	return l_prec


def label_based_macro_recall(y_true, y_pred):
    
    # compute true positive along axis = 0 i.e labels
    l_recall_num = np.sum(np.logical_and(y_true, y_pred), axis = 0)

    # compute true positive + false negatives along axis = 0 i.e columns
    l_recall_den = np.sum(y_true, axis = 0)

    # compute recall per class/label
    l_recall_per_class = l_recall_num/l_recall_den

    # compute macro averaged recall i.e recall averaged across labels. 
    l_recall = np.mean(l_recall_per_class)
    return l_recall


def label_based_micro_accuracy(y_true, y_pred):
    
    # sum of all true positives across all examples and labels 
    l_acc_num = np.sum(np.logical_and(y_true, y_pred))

    # sum of all tp+fp+fn across all examples and labels.
    l_acc_den = np.sum(np.logical_or(y_true, y_pred))

    # compute mirco averaged accuracy
    return l_acc_num/l_acc_den



def label_based_micro_precision(y_true, y_pred):
    
    # compute sum of true positives (tp) across training examples
    # and labels. 
    l_prec_num = np.sum(np.logical_and(y_true, y_pred))

    # compute the sum of tp + fp across training examples and labels
    l_prec_den = np.sum(y_pred)

    # compute micro-averaged precision
    return l_prec_num/l_prec_den


def label_based_micro_recall(y_true, y_pred):
	
    # compute sum of true positives across training examples and labels.
    l_recall_num = np.sum(np.logical_and(y_true, y_pred))
    # compute sum of tp + fn across training examples and labels
    l_recall_den = np.sum(y_true)

    # compute mirco-average recall
    return l_recall_num/l_recall_den


def compute_all_metrics(y_true, y_pred):

    emr_value = emr(y_true, y_pred)
    print('Exact Match Ratio : %0.2f %%'%(emr_value*100))

    hl_value = hamming_loss(y_true, y_pred)
    print('Hamming Loss : %0.2f %%'%(hl_value*100))

    ex_based_accuracy = example_based_accuracy(y_true, y_pred)
    print('Example Based Accuracy : %0.2f %%'%(ex_based_accuracy*100))

    ex_based_precision = example_based_precision(y_true, y_pred)
    print('Example Based Precision : %0.2f %%'%(ex_based_precision*100))

    lb_macro_acc_val = label_based_macro_accuracy(y_true, y_pred)
    print('Label Based Macro Accuracy : %0.2f %%'%(lb_macro_acc_val*100))

    lb_macro_precision_val = label_based_macro_precision(y_true, y_pred) 
    print('Label Based Macro Precision : %0.2f %%'%(lb_macro_precision_val*100))

    lb_macro_recall_val = label_based_macro_recall(y_true, y_pred) 
    print('Label Based Macro Recall : %0.2f %%'%(lb_macro_recall_val*100))

    lb_micro_acc_val = label_based_micro_accuracy(y_true, y_pred)
    print('Label Based Micro Accuracy : %0.2f %%'%(lb_micro_acc_val*100))

    lb_micro_prec_val = label_based_micro_precision(y_true, y_pred)
    print('Label Based Micro Precision : %0.2f %%'%(lb_micro_prec_val*100))

    lb_micro_recall_val = label_based_micro_recall(y_true, y_pred)
    print('Label Based Micro Recall : %0.2f %%'%(lb_micro_recall_val*100))