import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn import metrics
from sklearn import preprocessing
import os
import pandas as pd
from openpyxl import load_workbook
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle



MEASURES = ['balanced_accuracy', 'accuracy', 'sensitivity', 'specificity', 'precision', 'f1']


def evaluation(ground_truth, confidences, classes, outfile, sheet_name, row_names, permutation_test=None, ax=None, showit=True):

    C = len(classes)
    N = len(ground_truth)

    # probabilities scores to predictions
    prediction = np.argmax(confidences, axis=1)

    # column names
    column_names = ['ground truth', 'prediction'] + ['class {}: {}'.format(ci, c) for ci, c in enumerate(classes)] + ['classes'] + MEASURES + ['AUC']
    if permutation_test is not None:
        column_names += ['permutation test: score', 'permutation test: pvalue']

    # collect results in np array
    values = np.empty((N, len(column_names)))
    values[:] = np.nan
    values[:, 0] = ground_truth
    values[:, 1] = prediction
    values[:, 2:2+C] = confidences

    # Mean performances
    values_overall = performance_scores(ground_truth, prediction, average='macro')
    # Performance for each class
    values_classes = binary_multiclass_performance(ground_truth, prediction)
    # ROC curve
    auc = roc_curves(ground_truth, confidences, classes, pfile=outfile.replace('.xlsx', '_{}_roc.png'.format(sheet_name)), plotit=False, ax=ax)
    
    
    values[1:C+1, C+2] = range(0, C)
    values[0, C+3:C+9] = np.around(values_overall, decimals=3)
    values[1:1+C, C+3:C+9] = np.around(values_classes, decimals=3)
    values[0, C+9] = np.around(auc['macro'], decimals=3)
    values[1:1+C, C+9] = np.around(list(auc.values())[:C], decimals=3)

    if permutation_test is not None:
        values[0, -2:] = permutation_test

    # create dataframe
    d = {'ids': range(0, N)}
    df = pd.DataFrame(data=d)

    df['specimen'] = row_names
    # insert content with column names
    for m in range(0, len(column_names)):
        df[column_names[m]] = values[:, m].tolist()

    df_save_to_excel(outfile, df, sheet_name)

    return df

    

def df_save_to_excel(outfile, df, sheet_name):

    if os.path.isfile(outfile):
        book = load_workbook(outfile)
        writer = pd.ExcelWriter(outfile, engine='openpyxl')
        writer.book = book

    if os.path.isfile(outfile):
        df.to_excel(writer, sheet_name=sheet_name)
        writer.save()
        writer.close()
    else:
        df.to_excel(outfile, sheet_name=sheet_name)


def specificity_score(gt, pred, average='binary', pos_label=1):
    '''
        computes the specificity or true negative rate.
        Specificity = TN/N = TN/(TN + FP)

        Same behavior as metrics.precision_score, metrics.recall_score and metrics.f1_score
    '''

    confusion_matrix = metrics.multilabel_confusion_matrix(gt, pred)

    tp_sum = confusion_matrix[:, 1, 1]
    true_sum = tp_sum + confusion_matrix[:, 1, 0]
    tn_sum = confusion_matrix[:, 0, 0]
    false_sum = tn_sum + confusion_matrix[:, 0, 1]

    if average == 'micro':
        tn_sum = np.array([tn_sum.sum()])
        false_sum = np.array([false_sum.sum()])
    
    specificity = tn_sum / false_sum

    if average == 'binary':
        specificity = specificity[pos_label]
    elif average == 'macro':
        specificity = np.mean(specificity)
    elif average == 'weighted':
        specificity = np.average(specificity, weights=true_sum)


    return specificity


def performance_scores(gt, pred, average='macro', pos_label=1):
    import warnings
    warnings.filterwarnings('ignore')

    '''
        Computes for all class in a binary or multiclass setting, 
        the classification performances.
    '''

    # #
    # # Accuracy
    # #
    # d = np.trace(cfm)
    # acc = d / len(gt)
    # #
    # # Balanced Accuracy
    # #
    # d = 0
    # for i in range(0, NC):
    #     d = d + cfm[i, i] / np.sum(cfm[i, :])
    # bacc = d / NC

    assert average in ['binary', 'micro', 'macro', 'weighted']
    specificity = specificity_score(gt, pred, average=average, pos_label=pos_label)
    accuracy = metrics.accuracy_score(gt, pred)
    balanced_accuracy = metrics.balanced_accuracy_score(gt, pred)
    f1 = metrics.f1_score(gt, pred, average=average, pos_label=pos_label)
    sensitivity = metrics.recall_score(gt, pred, average=average, pos_label=pos_label)
    precision = metrics.precision_score(gt, pred, average=average, pos_label=pos_label)

    return np.asarray([balanced_accuracy, accuracy, sensitivity, specificity, f1, precision])


def binary_multiclass_performance(ground_truth, prediction):
    import warnings
    warnings.filterwarnings('ignore')

    '''
        Computes for each class individually in a binary or multiclass setting, 
        the classification performances.
    '''
    classes = set(ground_truth)

    values = -1*np.ones((len(classes), 6))
    # gt = preprocessing.label_binarize(ground_truth, classes=list(classes))
    # pred = preprocessing.label_binarize(prediction, classes=list(classes))
    for ci, c in enumerate(classes):
        gt = [1 if g==c else 0 for g in ground_truth]
        pred = [1 if p==c else 0 for p in prediction]
        vals = performance_scores(gt, pred, average='binary', pos_label=1)
        # vals = performance_scores(gt[:,ci], pred[:,ci], average='binary', pos_label=1)

        values[ci, :] = vals
    return values


def _test_example():

    ygt = [0,0,0,0,0,2,2,1,1,1]
    ypred = [0,0,0,0,2,2,0,0,0,1]
    classes = ['Null', 'Eins', 'Zwei']
    class_ids = set(ygt)

    measures = MEASURES
    values = binary_multiclass_performance(ygt, ypred)

    for ci in class_ids:
        print(classes[ci])
        for mi, m in enumerate(measures):
            print("{} ({:1.2f})".format(m, values[ci, mi]))
        print()


def roc_curves(ground_truth, prediction, classes, pfile=None, plotit=True, ax=None):

    C = len(classes)
    N = len(ground_truth)
    classes_num = set(ground_truth)

    # Binarize the output
    if C==2:
        ground_truth_bin = np.zeros((N, 2))
        ground_truth_bin[:,0] = np.array([0 if g==1 else 1 for g in ground_truth])
        ground_truth_bin[:,1] = ground_truth
        ground_truth = ground_truth_bin
    else:
        ground_truth = preprocessing.label_binarize(ground_truth, classes=list(classes_num))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(C):
        fpr[i], tpr[i], _ = metrics.roc_curve(ground_truth[:, i], prediction[:, i],)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(ground_truth.ravel(), prediction.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])


    '''
        Plot ROC curves for the multilabel problem
    '''

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(C)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(C):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= C

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    if ax is None:
        # plt.figure(figsize=[5,4])
        fig, ax = plt.subplots(figsize=(5,4))
    lw = 2
    ax.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    ax.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['red', 'green', 'blue', 'orange'])
    for i, color in zip(range(C), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve')
    ax.legend(loc="best", fontsize=6)
    

    if pfile is not None:
        plt.savefig(pfile, dpi = 300)
    if plotit:
        plt.show()

    return roc_auc 