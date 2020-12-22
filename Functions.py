from sklearn.metrics import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_classifiers(classifiers, x_test, y_test):

    result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

    for cls in classifiers:
        fpr, tpr, _ = roc_curve(y_test,  cls.predict_proba(x_test)[:, 1] )
        auc = roc_auc_score(y_test, cls.predict_proba(x_test)[:, 1] )
        result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                                'fpr':fpr, 
                                                'tpr':tpr, 
                                                'auc':auc}, ignore_index=True)

    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'], 
                 result_table.loc[i]['tpr'], 
                 label="{}, AUC={:.3f}".format(result_table.loc[i]['classifiers'], result_table.loc[i]['auc']))


    plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')


    plt.show()