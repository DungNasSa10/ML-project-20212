import pandas as pd

from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, auc,
    plot_confusion_matrix, plot_roc_curve, f1_score
)


def print_score(true, pred, train=True):
    if train:
        clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
        res =  ("Train Result:\n================================================\n" 
               f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%\n" 
                "_______________________________________________\n" 
               f"CLASSIFICATION REPORT:\n{clf_report}\n" 
                "_______________________________________________\n" 
               f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")
        
    elif train==False:
        clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
        res =  ("Test Result:\n================================================\n"       
               f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%\n" 
                "_______________________________________________\n" 
               f"CLASSIFICATION REPORT:\n{clf_report}\n" 
                "_______________________________________________\n" 
               f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")
               
    return res


def print_dict(info: dict):
    print()
    for k, v in info.items():
        print(k, "=", v)
    print()