import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits

digits = load_digits()

X = digits.data 
Y = digits.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
Y_pred_prob = knn.predict_proba(X_test)

Y_pred_train = knn.predict(X_train)
Y_pred_prob_train = knn.predict_proba(X_train)

acc_train = accuracy_score(Y_train, Y_pred_train)
lloss_train = log_loss(Y_train, Y_pred_prob_train)
conf_train = confusion_matrix(Y_train, Y_pred_train)

acc = accuracy_score(Y_test, Y_pred)
lloss = log_loss(Y_test, Y_pred_prob)
conf = confusion_matrix(Y_test, Y_pred)

print(f'TEST - Accuracy: {acc}, Loss: {lloss}')
print(conf)
print('--' * 30)
print(f'TRAIN - Accuracy: {acc_train}, Loss: {lloss_train}')
print(conf_train)


# provo con più valori di K

Ks = [1,2,3,4,5,6,7,8,9,10]

print('--' * 30)
print('--' * 30)

for k in Ks:
    print('K = ' + str(k))
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    Y_pred_prob = knn.predict_proba(X_test)

    Y_pred_train = knn.predict(X_train)
    Y_pred_prob_train = knn.predict_proba(X_train)

    acc_train = accuracy_score(Y_train, Y_pred_train)
    lloss_train = log_loss(Y_train, Y_pred_prob_train)
    conf_train = confusion_matrix(Y_train, Y_pred_train)

    acc = accuracy_score(Y_test, Y_pred)
    lloss = log_loss(Y_test, Y_pred_prob)
    conf = confusion_matrix(Y_test, Y_pred)

    print(f'TEST - Accuracy: {acc}, Loss: {lloss}')
    print(f'TRAIN - Accuracy: {acc_train}, Loss: {lloss_train}')

    print('--' * 30)
    print('--' * 30)


# visulaizzazione modello per K = 3
    
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)

import matplotlib.pyplot as plt

for i in range(len(X_test)):
    if(Y_pred[i] != Y_test[i]):
        print(f'Numero {Y_test[i]} classificato come {Y_pred[i]}')
        plt.imshow(X_test[i].reshape([8,8]), cmap="gray")
        plt.show()