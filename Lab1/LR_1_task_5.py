import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

df = pd.read_csv('data_metrics.csv')
df.head()

thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
df.head()

print(confusion_matrix(df.actual_label.values, df.predicted_RF.values))

# Функція для обчислення кількості істинних позитивних
def bondar_find_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))

# Функція для обчислення кількості помилкових негативних
def bondar_find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))

# Функція для обчислення кількості помилкових позитивних
def bondar_find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))

# Функція для обчислення кількості істинних негативних
def bondar_find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))

print('TP:',bondar_find_TP(df.actual_label.values, df.predicted_RF.values))
print('FN:',bondar_find_FN(df.actual_label.values, df.predicted_RF.values))
print('FP:',bondar_find_FP(df.actual_label.values, df.predicted_RF.values))
print('TN:',bondar_find_TN(df.actual_label.values, df.predicted_RF.values))


def find_conf_matrix_values(y_true,y_pred):
    # calculate TP, FN, FP, TN
    TP = bondar_find_TP(y_true,y_pred)
    FN = bondar_find_FN(y_true,y_pred)
    FP = bondar_find_FP(y_true,y_pred)
    TN = bondar_find_TN(y_true,y_pred)
    return TP,FN,FP,TN

def bondar_confusion_matrix(y_true, y_pred):
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
    return np.array([[TN,FP],[FN,TP]])

print(bondar_confusion_matrix(df.actual_label.values, df.predicted_RF.values))

assert np.array_equal(bondar_confusion_matrix(df.actual_label.values, df.predicted_RF.values),
                      confusion_matrix(df.actual_label.values, df.predicted_RF.values)), \
    print('bondar_confusion_matrix() is not correct for RF')

# Виклик функції для моделі LR
assert np.array_equal(bondar_confusion_matrix(df.actual_label.values, df.predicted_LR.values),
                      confusion_matrix(df.actual_label.values, df.predicted_LR.values)), \
    print('bondar_confusion_matrix() is not correct for LR')

#оцінка точності
print(accuracy_score(df.actual_label.values, df.predicted_RF.values))

def bondar_accuracy_score(y_true, y_pred):
# calculates the fraction of samples
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
    return (TP + TN) / (TP + TN + FP + FN)
# Тестування для моделі RF
assert bondar_accuracy_score(df.actual_label.values, df.predicted_RF.values) == accuracy_score(df.actual_label.values, df.predicted_RF.values), 'bondar_accuracy_score failed on RF'

# Тестування для моделі LR
assert bondar_accuracy_score(df.actual_label.values, df.predicted_LR.values) == accuracy_score(df.actual_label.values, df.predicted_LR.values), 'bondar_accuracy_score failed on LR'

print('Accuracy RF: %.3f' % bondar_accuracy_score(df.actual_label.values, df.predicted_RF.values))
print('Accuracy LR: %.3f' % bondar_accuracy_score(df.actual_label.values, df.predicted_LR.values))

#RECALL----------------------------
recall_score(df.actual_label.values, df.predicted_RF.values)


def bondar_recall_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    recall = TP / (TP + FN)
    return recall

# Тестування для моделі RF
assert bondar_recall_score(df.actual_label.values, df.predicted_RF.values) == recall_score(df.actual_label.values,
                                                                                           df.predicted_RF.values), 'bondar_recall_score failed on RF'
# Тестування для моделі LR
assert bondar_recall_score(df.actual_label.values, df.predicted_LR.values) == recall_score(df.actual_label.values,
                                                                                           df.predicted_LR.values), 'bondar_recall_score failed on LR'
# Виведення recall для моделі RF
print('Recall RF: %.3f' % bondar_recall_score(df.actual_label.values, df.predicted_RF.values))

# Виведення recall для моделі LR
print('Recall LR: %.3f' % bondar_recall_score(df.actual_label.values, df.predicted_LR.values))

#PRECISION
precision_score(df.actual_label.values, df.predicted_RF.values)


def bondar_precision_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    precision = TP / (TP + FP)
    return precision

# Тестування для моделі RF
assert bondar_precision_score(df.actual_label.values, df.predicted_RF.values) == precision_score(df.actual_label.values,
                                                                                                 df.predicted_RF.values), 'bondar_precision_score failed on RF'
# Тестування для моделі LR
assert bondar_precision_score(df.actual_label.values, df.predicted_LR.values) == precision_score(df.actual_label.values,
                                                                                                 df.predicted_LR.values), 'bondar_precision_score failed on LR'
# Виведення точності для моделі RF
print('Precision RF: %.3f' % bondar_precision_score(df.actual_label.values, df.predicted_RF.values))
# Виведення точності для моделі LR
print('Precision LR: %.3f' % bondar_precision_score(df.actual_label.values, df.predicted_LR.values))

#f1_score------------------------
f1_score(df.actual_label.values, df.predicted_RF.values)


def bondar_f1_score(y_true, y_pred):
    recall = bondar_recall_score(y_true, y_pred)
    precision = bondar_precision_score(y_true, y_pred)
    f1 = (2*(precision * recall)) / (precision + recall)
    return f1

# Тестування для моделі RF
assert bondar_f1_score(df.actual_label.values, df.predicted_RF.values) == f1_score(df.actual_label.values,
                                                                                   df.predicted_RF.values), 'bondar_f1_score failed on RF'
# Тестування для моделі LR

try:
    assert np.array_equal(bondar_f1_score(df.actual_label.values, df.predicted_LR.values),
                          f1_score(df.actual_label.values,df.predicted_LR.values))
    print("bondar_confusion_matrix() is correct for LR")
except AssertionError:
    print("bondar_confusion_matrix() is not correct for LR")

# Виведення F1-міри для моделі RF
print('F1 RF: %.3f' % bondar_f1_score(df.actual_label.values, df.predicted_RF.values))
# Виведення F1-міри для моделі LR
print('F1 LR: %.3f' % bondar_f1_score(df.actual_label.values, df.predicted_LR.values))

#ПОРОГИ

print('scores with threshold = 0.5')
print('Accuracy RF: %.3f' % bondar_accuracy_score(df.actual_label.values, df.predicted_RF.values))
print('Recall RF: %.3f' % bondar_recall_score(df.actual_label.values, df.predicted_RF.values))
print('Precision RF: %.3f' % bondar_precision_score(df.actual_label.values, df.predicted_RF.values))
print('F1 RF: %.3f' % bondar_f1_score(df.actual_label.values, df.predicted_RF.values))

print('\nScores with threshold = 0.25')
predicted_RF_025 = (df.model_RF >= 0.25).astype('int').values
print('Accuracy RF: %.3f' % bondar_accuracy_score(df.actual_label.values, predicted_RF_025))
print('Recall RF: %.3f' % bondar_recall_score(df.actual_label.values, predicted_RF_025))
print('Precision RF: %.3f' % bondar_precision_score(df.actual_label.values, predicted_RF_025))
print('F1 RF: %.3f' % bondar_f1_score(df.actual_label.values, predicted_RF_025))

#roc_curve та roc_auc_score
fpr_RF, tpr_RF,thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF')
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR')
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

#roc_auc_score

auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print('AUC RF:%.3f'% auc_RF)
print('AUC LR:%.3f'% auc_LR)

plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF AUC: %.3f'%auc_RF)
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR AUC: %.3f'%auc_LR)
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()