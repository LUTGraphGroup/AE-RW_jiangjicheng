import warnings
import time
from keras import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adadelta
from keras.utils import to_categorical
from matplotlib import pyplot, pyplot as plt
from numpy import interp
warnings.filterwarnings("ignore")
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, auc, roc_curve, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from xgboost import XGBClassifier
RW_feature = np.loadtxt('node_features.txt')
AE_feature = np.loadtxt('features11.txt')

def get_data(data_path):
    total_Sample = []
    Lable = []
    with open(data_path) as f:
        for line in f:
            item = line.strip().split()
            mirna = int(item[0])
            disease = int(item[1])
            lable = int(item[2])
            Lable.append(lable)
            feature_vec = RW_feature[mirna] + AE_feature[disease].tolist()
            total_Sample.append(feature_vec)
    total_Sample.reverse()
    Lable.reverse()
    return total_Sample, Lable

def get_train_data():
    # data_path = 'train_data.txt'
    # data_path = 'test_data.txt'
    data_path = 'test-order.txt'
    total_sample, label = get_data(data_path)
    return total_sample, label

def DNN():
    model = Sequential()
    model.add(Dense(input_dim=128, output_dim=500))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(input_dim=500, output_dim=500, init='glorot_normal'))
    model.add(Dropout(0.1))

    model.add(Dense(input_dim=500, output_dim=300))#,init='glorot_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(input_dim=300, output_dim=2))#,init='glorot_normal'))
    model.add(Activation('sigmoid'))
    # sgd = SGD(lr=0.03, decay=1e-6, momentum=0.9, nesterov=True)
    # adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adadelta)
    return model

def train(X, Y):
    X, Y = shuffle(X, Y, random_state=200)
    # clf = XGBClassifier(n_estimators=1000, learning_rate=0.1, random_state=200)
    #对比实验分类器
    # from sklearn.ensemble import RandomForestClassifier
    # clf = RandomForestClassifier(n_estimators=1000, random_state=200)
    # from sklearn.linear_model import LogisticRegression
    # clf = LogisticRegression(random_state=200)
    # from sklearn.naive_bayes import GaussianNB
    # clf = GaussianNB()
    # from sklearn.svm import SVC
    # clf = SVC(probability=True)
    # from sklearn.neighbors import KNeighborsClassifier
    # clf = KNeighborsClassifier(n_neighbors=5)
    # from sklearn.tree import DecisionTreeClassifier
    # clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    model_DNN = DNN()

    kf = KFold(n_splits=5)
    # kf = KFold(n_splits=10)
    print('-----------------------------------------------')
    t = 0
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    AUC_list = []
    AUPRC_list = []
    Accuracy_list = []
    Precision_list = []
    Recall_list = []
    F1_list = []
    # 在循环外部创建一个空列表
    all_predict_values = []

    for train_index, test_index in kf.split(X, Y):
        X_train = X[train_index]
        Y_train = Y[train_index]
        X_test = X[test_index]
        Y_test = Y[test_index]

        # Convert labels to one-hot vectors
        Y_train = to_categorical(Y_train)
        Y_test = to_categorical(Y_test)

        # Y_train = np.argmax(Y_train, axis=1)
        # Y_test = np.argmax(Y_test, axis=1)
        model_DNN.fit(X_train, Y_train, batch_size=200, nb_epoch=50, shuffle=True)
        predict_value = model_DNN.predict_proba(X_test, batch_size=200, verbose=True)[:, 1]
        # # 将每一折的预测值追加到列表中
        # all_predict_values.append(predict_value)
        # # 将列表中的数组合并为一个更大的数组
        # all_predict_values = np.concatenate(all_predict_values)
        # # 将整个列表保存到文件中
        # np.savetxt("predict_values.txt", all_predict_values, delimiter="\t", fmt='%.4f')

        # 在循环内部使用不同的文件名保存每一折的预测值
        # filename = f"predict_values_fold{kf}.txt"
        # np.savetxt(filename, predict_value, delimiter="\t", fmt='%.4f')
        # clf.fit(X_train, Y_train)
        # predict_value = clf.predict_proba(X_test)[:, 1]
        # AUC = metrics.roc_auc_score(Y_test[:, 1], predict_value)
        # predict_value = clf.predict_proba(X_test)[:, 1]
        # AUC = metrics.roc_auc_score(Y_test, predict_value)
        # AUC_list.append(AUC)
        # fpr, tpr, auc_thresholds = roc_curve(Y_test, predict_value)
        # auc_score = auc(fpr, tpr)
        # Precision, Recall, _ = precision_recall_curve(Y_test, predict_value)
        # AUPRC = auc(Recall, Precision)
        # AUPRC_list.append(AUPRC)
        #
        # #其他指标
        # A = accuracy_score(Y_test, predict_value.round())
        # Accuracy_list.append(A)
        # P = precision_score(Y_test, predict_value.round())
        # Precision_list.append(P)
        # R = recall_score(Y_test, predict_value.round())
        # Recall_list.append(R)
        # F1 = f1_score(Y_test, predict_value.round())
        # F1_list.append(F1)

        AUC = metrics.roc_auc_score(Y_test[:, 1], predict_value)
        AUC_list.append(AUC)
        fpr, tpr, auc_thresholds = roc_curve(Y_test[:, 1], predict_value)
        auc_score = auc(fpr, tpr)
        Precision, Recall, _ = precision_recall_curve(Y_test[:, 1], predict_value)
        AUPRC = auc(Recall, Precision)
        AUPRC_list.append(AUPRC)

        # 其他指标
        A = accuracy_score(Y_test[:, 1], predict_value.round())
        Accuracy_list.append(A)
        P = precision_score(Y_test[:, 1], predict_value.round())
        Precision_list.append(P)
        R = recall_score(Y_test[:, 1], predict_value.round())
        Recall_list.append(R)
        F1 = f1_score(Y_test[:, 1], predict_value.round())
        F1_list.append(F1)

        # precision1, recall, pr_threshods = precision_recall_curve(Y_test[:, 1], predict_value)
        # aupr_score = auc(recall, precision1)
        t = t + 1
        np.savetxt("AE-RW_fpr_" + str(t) + ".txt", fpr)
        np.savetxt("AE-RW_tpr_" + str(t) + ".txt", tpr)
        # 在循环内部使用不同的文件名保存每一折的预测值
        filename = f"predict_values_fold{t}.txt"
        np.savetxt(filename, predict_value, delimiter="\t", fmt='%.4f')
        pyplot.plot(fpr, tpr, linewidth=1, alpha=0.5, label='ROC fold %d (AUC = %0.4f)' % (t, auc_score))
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
    mean_tpr /= kf.n_splits
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    pyplot.plot(mean_fpr, mean_tpr, 'black', linewidth=1.5, alpha=0.8,
                label='Mean ROC(AUC = %0.4f)' % mean_auc)

    pyplot.legend()

    plt.savefig('5-fold CV AE-RW(AUC = %0.4f).png' % mean_auc, dpi=300)
    pyplot.show()
    #保存每一折的fpr和tpr数据
    np.savetxt('mean_fpr.csv', mean_fpr, delimiter=',')
    np.savetxt('mean_tpr.csv', mean_tpr, delimiter=',')


if __name__ == "__main__":

    sample_data, lable = get_train_data()
    sample_data = np.array(sample_data)
    lable = np.array(lable)


    time_start = time.time()
    train(sample_data, lable)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
































