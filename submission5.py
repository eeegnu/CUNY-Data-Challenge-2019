import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import MiniBatchSparsePCA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import fastica
import sklearn.linear_model as sklm
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

def concat(lsts):
    a = []
    for i in lsts:
        a.extend(i)
    return a

def runTest(clf,mf, showGraph = False, upper = 1, lower = 0):
    global test_solution
    clf.fit(x_train0[mf], x_train0.passed)
    test_solution = clf.predict_proba(x_test0[mf])
    for i in range(len(test_solution)):
        if test_solution[i][0] < lower:
            test_solution[i][0] = lower
            test_solution[i][1] = 1-lower
        elif test_solution[i][0] > upper:
            test_solution[i][0] = upper
            test_solution[i][1] = 1-upper
    print(log_loss(x_test0.passed.values, test_solution))

    total= 0
    correct = 0

    for i in range(len(x_test0.passed.values)):
        if x_test0.passed.values[i] == round(test_solution[i][1]):
            correct += 1
        total += 1
    print(correct / total)

    if showGraph:
        errorVisualizer(test_solution,100)

def runCTest(clf,mf):
    global test_solution
    clf.fit(x_train0[mf], x_train0.passed)
    test_solution = clf.predict(x_test0[mf])
    try:
        print('accuracy:',accuracy_score(x_test0.passed.values, test_solution.round()),type(clf))
    except:
        print('score:', clf.score(x_test0[mf],x_test0.passed.values),type(clf))

    
def runTestDirect(clf,train,test,showGraph = False):
    clf.fit(train.drop(columns = ['passed']), train['passed'])
    test_solution = clf.predict_proba(test.drop(columns = ['passed']))
    print(log_loss(test['passed'].values, test_solution))

    total= 0
    correct = 0

    for i in range(len(test['passed'].values)):
        if test['passed'].values[i] == round(test_solution[i][1]):
            correct += 1
        total += 1
    print(correct / total)

    if showGraph:
        errorVisualizer(test_solution,1000)

def runTestTransformed(clf, train, test, showGraph = False):
    clf.fit(train, x_train0.passed)
    test_solution = clf.predict_proba(test)
    print(log_loss(x_test0.passed.values, test_solution))

    total= 0
    correct = 0

    for i in range(len(x_test0.passed.values)):
        if x_test0.passed.values[i] == round(test_solution[i][1]):
            correct += 1
        total += 1
    print(correct / total)

    if showGraph:
        errorVisualizer(test_solution,100)

def calcAcc(truth,probs):
    total = 0
    for i in range(len(probs)):
        total += abs(truth[i] - np.round(probs[i][0]))
    return total / len(probs)

def errorVisualizer(probs,bins):
    fig, axs = plt.subplots(4)
    probBins = [[] for _ in range(bins+1)]
    amounts = [0]*(bins)
    accuracy = [0]*(bins)
    ind = 0
    for j in probs:
        probBins[int(j[1]*bins)].append(ind)
        ind += 1
    errors = [0]*bins
    for j in range(bins):
        if len(probBins[j]) == 0:
            continue
        if len(x_test0.passed.values[probBins[j]]) > 0 and (min(x_test0.passed.values[probBins[j]]) == 1 or max(x_test0.passed.values[probBins[j]]) == 0):
            continue
        errors[j] = log_loss(x_test0.passed.values[probBins[j]], probs[probBins[j]])
        amounts[j] = len(x_test0.passed.values[probBins[j]])
        accuracy[j] = calcAcc(x_test0.passed.values[probBins[j]],probs[probBins[j]])
    
    axs[0].plot(np.arange(0,1,1/bins),errors)
    axs[1].plot(np.arange(0,1,1/bins),amounts)
    axs[2].plot(np.arange(0,1,1/bins),accuracy)
    axs[3].plot(np.arange(0,1,1/bins),[amounts[q]*errors[q]/(len(x_test0.passed.values)) for q in range(len(amounts))])
    print(sum([amounts[q]*errors[q]/(len(x_test0.passed.values)) for q in range(len(amounts))]))
    plt.show()

    fig, axs = plt.subplots(2)
    axs[0].plot(np.arange(0,1,1/bins),amounts)
    axs[1].plot(np.arange(0,1,1/bins),[amounts[q]*errors[q]/(len(x_test0.passed.values)) for q in range(len(amounts))])
    plt.show()

d = pd.read_csv('All_Features.csv', parse_dates=['inspection_date'])
fin = pd.read_csv('All_Features_Final.csv', parse_dates=['inspection_date'])
#d = d.replace(np.nan, None, regex=True)
x_train0 = d
x_test0 = d

#x_train0, x_test0 = train_test_split(d, test_size=0.2)

#Data mixed for matlab:
#x_train0 = pd.read_csv(r'C:\Users\Eugene\Documents\Kaggle\DataForMatlab\train.csv')
#x_test0 = pd.read_csv(r'C:\Users\Eugene\Documents\Kaggle\DataForMatlab\test.csv')

#Generate features we can choose from:

violations = ['vBin' + str(i) for i in range(92)]
viols = ['vBin' + str(i) for i in range(92)]
inspections = ['iBin' + str(i) for i in range(33)]
actions = ['aBin' + str(i) for i in range(6)]

PassFail = ['PassedPrior','FailedPrior', 'priorResult']
VenueStats = ['boro', 'street', 'cuisine_description', 'dba_counts']
SumsF = ['SimpleGuess','s1','s2','s3','s4']
CuisineSum = ['cuis_sum']
tax = ['lat', 'long', 'population']
taxBrackets = []
weather = ['TMAX','TMIN', 'TAVG', 'PRCP']


#Begin the (many) trials!
every = concat([violations,inspections,actions,PassFail,VenueStats,SumsF,weather])
test = concat([violations])

clf1 = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=10)
clf2 = ExtraTreesClassifier(n_estimators=50, max_depth=10, min_samples_leaf=10)
clf3 = HistGradientBoostingClassifier(l2_regularization = 1, min_samples_leaf = 17, max_iter = 215)
clf4 = HistGradientBoostingClassifier(l2_regularization = 1)
clf5 = KNeighborsClassifier(n_neighbors=20)
clf6 = DecisionTreeClassifier(splitter = 'random', min_samples_split = 20)

runTest(clf1,every)
runTest(clf2,every)
runTest(clf3,every,True)
runTest(clf4,every)
runTest(clf6,every,lower = 0.01,upper = 0.99)

sub = clf4.predict_proba(fin[every])

# take just the `id` and `n_violations` columns (since that's all we need)
submission = fin[['id']].copy()

tmp = []
for i in sub:
    if i[1] > 1:
        tmp.append(0.99)
    elif i[1] < 0:
        tmp.append(0.01)
    else:
        tmp.append(i[1])
submission['Predicted'] = tmp

# IMPORTANT: Kaggle expects you to name the columns `Id` and `Predicted`, so let's make sure here
submission.columns = ['Id', 'Predicted']

# write the submission to a csv file so that we can submit it after running the kernel
#submission.to_csv('submission5.csv', index=False)








