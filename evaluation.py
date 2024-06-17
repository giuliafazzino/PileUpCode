import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.utils import class_weight


params = {
    'font.size': 15
}
plt.rcParams.update(params)

def plot_proba(labels_train, preds_train, labels_test, preds_test, norm, fig_name):
    fig, ax = plt.subplots(figsize = [12,6])

    # Train
    _, bins, _ = ax.hist(preds_train[labels_train == 0], histtype ='step', color = 'sandybrown',  
            linewidth = 2, label = 'Hard scatter & pile-up (train)', density = norm, bins = 20)
    
    ax.hist(preds_train[labels_train == 1], histtype ='step', color = 'darkred', 
            linewidth = 2, label = 'No pile-up (train)', density = norm, bins = bins)

    # Test
    ax.hist(preds_test[labels_test == 0], histtype ='step', color = 'lightskyblue',
            linewidth = 2, label = 'Hard scatter & pile-up (test)', density = norm, bins = bins)
    ax.hist(preds_test[labels_test == 1], histtype ='step', color = 'darkblue', 
            linewidth = 2, label = 'No pile-up (test)', density = norm, bins = bins)
    
    
    ax.set_xlabel('Probability to be a no pile-up cluster')
    plt.legend(ncol = 2)
    plt.savefig(fig_name)
    plt.close()
    print('Saving output in ' + fig_name + '\n')

    return



def roc(labels_train, preds_train, labels_test, preds_test, plot_path = 'out/plots'):
   
   
   # Make and plot ROC curve
   fpr_train, tpr_train, thresholds_train = metrics.roc_curve(labels_train, preds_train)
   roc_auc_train = metrics.roc_auc_score(labels_train, preds_train)

   fpr_test, tpr_test, thresholds_test = metrics.roc_curve(labels_test, preds_test)
   roc_auc_test = metrics.roc_auc_score(labels_test, preds_test)

   plt.title('ROC Curve')
   plt.plot(fpr_train, tpr_train, 'b', label='AUC (train) = %0.4f'% roc_auc_train)
   plt.plot(fpr_test, tpr_test, 'g', label='AUC (test) = %0.4f'% roc_auc_test)

  
   # Find optimal threshold from ROC
   best_index = np.argmax(tpr_test - fpr_test) 
   plt.plot(fpr_test[best_index], tpr_test[best_index], 'go', markersize = 10, markerfacecolor = 'white',
            label = f'Optimal working point : {thresholds_test[best_index]:.2f}')


   plt.legend(loc='lower right')
   plt.plot([0,1],[0,1],'r--')
   plt.xlim([0,1])
   plt.ylim([0,1])
   plt.ylabel('True Positive Rate')
   plt.xlabel('False Positive Rate')

   print(f'Optimal threshold from ROC = {thresholds_test[best_index]:.2f}')
   plt.savefig(plot_path + '/ROC.pdf')
   plt.close()
   print('Saving output in ' + plot_path + '/ROC.pdf \n')

   #return thresholds_test[best_index]
   return

def pr(labels_train, preds_train, labels_test, preds_test, plot_path = 'out/plots'):
   
   
   # Make and plot PR curve
   pre_train, rec_train, thresholds_train = metrics.precision_recall_curve(labels_train, preds_train)
   ap_train = metrics.average_precision_score(labels_train, preds_train)

   pre_test, rec_test, thresholds_test = metrics.precision_recall_curve(labels_test, preds_test)
   ap_test = metrics.average_precision_score(labels_test, preds_test)

   plt.title('PR Curve')
   plt.plot(rec_train, pre_train, 'b', label='Average precision (train) = %0.4f'% ap_train)
   plt.plot(rec_test, pre_test, 'g', label='Average precision (test) = %0.4f'% ap_test)

   plt.legend(loc='lower right')
   plt.xlim([0,1])
   plt.ylim([0,1])
   plt.ylabel('Precision')
   plt.xlabel('Recall')

   plt.savefig(plot_path + '/PR.pdf')
   plt.close()
   print('Saving output in ' + plot_path + '/PR.pdf \n')

   return

def plot_confmat(labels_true, labels_pred, threshold, plot_path = 'out/plots'):
   confusion_mat = metrics.confusion_matrix(labels_true, labels_pred, normalize = 'true')
   fig = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
   fig.plot().ax_.set_title(f'Threshold = {threshold}')
   
   fig.figure_.savefig(plot_path + '/confusion_matrix.pdf')
   print('Saving output in ' + plot_path + '/confusion_matrix.pdf \n')

   return





def main():

    dir_path = 'out'
    plot_path = dir_path + '/plots'


    # Read files
    y_true_train = np.load(dir_path + '/trueClass_train.npy')
    probs_train = np.load(dir_path + '/predictions_train.npy')
    x_train = np.load(dir_path + '/x_train.npy')

    y_true_test = np.load(dir_path + '/trueClass_test.npy')
    probs_test = np.load(dir_path + '/predictions_test.npy')
    x_test = np.load(dir_path + '/x_test.npy')

    # Plot output probabilities
    print('Generating plot for probabilities ...')
    plot_proba(y_true_train, probs_train, y_true_test, probs_test, True, plot_path + '/probabilities_norm.pdf')

    # Plot ROC curve
    print('Generatig ROC curve ... ')
    roc(y_true_train, probs_train, y_true_test, probs_test, plot_path)

    # Plot PR curve
    print('Generatig PR curve ... ')
    pr(y_true_train, probs_train, y_true_test, probs_test, plot_path)    


    #####################################################
    #  Clusters outside collision time window (12.5 s)  #
    #####################################################
    
    # Masks 
    train_oot = np.abs(x_train[:,2]) >= 12.5**(1./3.)
    test_oot  = np.abs(x_test[:,2])  >= 12.5**(1./3.)

    # Plot timing distribution
    bins = np.arange(-72.5, 50.5, 5)

    plt.figure(figsize = [12,6])
    plt.hist((x_train[:,2])**3, histtype = 'step', label = 'Train', color = 'darkred', bins = bins, fill = False, log = True, linewidth = 3)
    plt.hist((x_test[:,2])**3, histtype = 'step', label = 'Test', color = 'darkblue', bins = bins, fill = False, log = True, linewidth = 3)

    plt.hist((x_train[:,2][train_oot])**3, histtype = 'stepfilled', label = 'Train (out of time)', color = 'sandybrown', 
             alpha = .7, bins = bins, log = True)
    plt.hist((x_test[:,2][test_oot])**3, histtype = 'stepfilled', label = 'Test (out of time)', color = 'lightskyblue', 
             alpha = .7, bins = bins, log = True)
    
    plt.xlabel('Time (ns)')
    plt.legend()
    plt.savefig('out/plots/timing.pdf')
    plt.close()

    # Plot probabilities
    print('Generating plot for probabilities for out of time clusters ...')
    plot_proba(y_true_train[train_oot], probs_train[train_oot], y_true_test[test_oot], probs_test[test_oot], 
               True, plot_path + '/probabilities_norm_oot.pdf')


    # Predict classes
    #y_pred = probs > threshold

    # Plot confusion matrix
    #print('Generatig confusion matrix ... ')
    #plot_confmat(y_true, y_pred, threshold, plot_path)

    return



if __name__ == '__main__':
    main()
    pass