import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Поділ даних на тренувальні та тестові вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Байєсівський класифікатор
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)
y_pred_nb = nb_clf.predict(X_test)

# Оцінка якості класифікації для Наївного Байєсівського Класифікатора
print("\nNaive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))
print(f"Accuracy Naive Bayes: {accuracy_score(y_test, y_pred_nb) * 100:.2f}%")

# SVM
svm_clf = SVC(kernel='linear')
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)

# Оцінка якості класифікації для SVM
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))
print(f"Accuracy SVM: {accuracy_score(y_test, y_pred_svm) * 100:.2f}%")

