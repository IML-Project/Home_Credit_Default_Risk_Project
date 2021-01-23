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


def basic_plots(train):
    # Find correlations with the target and sort
    train_correlations = train.corr(method="pearson")['TARGET'].sort_values()
    # Display correlations
    print(' Positive Correlations:\n', train_correlations.tail(15))
    print('\n Negative Correlations:\n', train_correlations.head(15))

    # loans that were paid off(value 0) and that weren't (value 1)
    train["TARGET"].value_counts();
    
    train["TARGET"].astype(int).plot.hist();
    plt.show();


    #unique classes from each column
    print(train.select_dtypes('object').apply(pd.Series.nunique, axis = 0))




    # Change negative to positive 
    (train["DAYS_BIRTH"]/-365).describe()
    

    # Show days of employment histogram
    train['DAYS_EMPLOYED'].describe()
    train['DAYS_EMPLOYED'].plot.hist(title="Days Employment Histogram");
    plt.show();



    
    
    plt.figure(figsize = (8, 6))

    # KDE plot of loans that were repaid on time
    sns.kdeplot(train.loc[train['TARGET'] == 0, 'DAYS_BIRTH'] / -365, label = 'target == 0')

     # KDE plot of loans which were not repaid on time
    sns.kdeplot(train.loc[train['TARGET'] == 1, 'DAYS_BIRTH'] / -365, label = 'target == 1')

    # Labeling of plot
    plt.xlabel('Age'); plt.ylabel('Density'); plt.title('Distribution of Ages');
    plt.legend();
    plt.show()



    

    # Age information into a separate dataframe
    age_data = train[['TARGET', 'DAYS_BIRTH']]
    age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / -365

    # Bin the age data
    age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
    age_data.head(10)

    # Group by the bin and calculate averages
    age_groups  = age_data.groupby('YEARS_BINNED').mean()
    

    
    plt.figure(figsize = (8, 8))
    # Graph the age bins and the average of the target as a bar plot
    plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])
    # Plot labeling
    plt.xticks(rotation = 75); plt.xlabel('Age Group '); plt.ylabel('Default Rate')
    plt.title('Default by Age Group');
    plt.show()



    extracted_data = train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]
    extracted_data_corrs = extracted_data.corr()



    
    plt.figure(figsize = (6, 6))
    # Heatmap of correlations
    sns.heatmap(extracted_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
    plt.title('Correlation Heatmap');
    plt.show()




    plt.figure(figsize = (10, 12))
    # iterate through the sources
    for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
        # create a new subplot for each source
        plt.subplot(3, 1, i + 1)
        # plot repaid loans
        sns.kdeplot(train.loc[train['TARGET'] == 0, source], label = 'target == 0')
        # plot loans that were not repaid
        sns.kdeplot(train.loc[train['TARGET'] == 1, source], label = 'target == 1')
    
        # Label the plots
        plt.title('Distribution of %s by Target Value' % source)
        plt.xlabel('%s' % source); plt.ylabel('Density');
    
    plt.tight_layout(h_pad = 2.5)
    plt.show()

