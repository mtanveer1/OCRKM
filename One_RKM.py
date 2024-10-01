import numpy as np
import time

def calculate_kA(A, C, mew):
    norm_A = np.linalg.norm(A, axis=1)[:, np.newaxis]
    norm_C = np.linalg.norm(C, axis=1)

    first_term = np.power(norm_A, 2)
    second_term = -2 * np.dot(A, C.T)
    third_term = np.power(norm_C, 2)

    kA = np.exp(-(1 / (2 * mew)) * (first_term + second_term + third_term))
    return kA

def Evaluate(ACTUAL, PREDICTED):
    idx = (ACTUAL == 1)
    p = np.sum(idx)
    n = np.sum(~idx)
    N = p + n
    tp = np.sum(np.logical_and(ACTUAL[idx] == 1, PREDICTED[idx] == 1))
    tn = np.sum(np.logical_and(ACTUAL[~idx] == 0, PREDICTED[~idx] == 0))
    fp = n - tn
    fn = p - tp
    tp_rate = tp / p if p != 0 else 0
    tn_rate = tn / n if n != 0 else 0
    accuracy = 100 * (tp + tn) / N
    sensitivity = 100 * tp_rate
    specificity = 100 * tn_rate
    precision = 100 * tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = sensitivity
    f_measure = 2 * ((precision * sensitivity) / (precision + sensitivity)) if (precision + sensitivity) != 0 else 0
    gmean = 100 * np.sqrt(tp_rate * tn_rate)
    
    EVAL = [accuracy, sensitivity, specificity, precision, recall, f_measure, gmean, tp, tn, fp, fn]
    return EVAL

def One_RKM(train_data, test_data, eta, lambd, mew):

    start = time.time()
    train=train_data[:,:-1]
    trainY=train_data[:,-1]

    etaInv = 1 / eta
    [m,n]=train.shape
    omega=calculate_kA(train, train, mew)
    mat1 = etaInv * omega + lambd*np.eye(m)
    e = np.ones((m, 1))
    deno = np.dot(e.T, np.linalg.solve(mat1, e))
    yh = np.linalg.solve(mat1, e) / deno
    rho = 1 / deno

    w=etaInv * np.dot(omega,yh)

    test=test_data[:,:-1]
    Y=test_data[:,-1]
    Y=Y.reshape(Y.shape[0], 1)
    K=calculate_kA(test, train, mew)

    # predicted_label=predict(K, w, rho)
    f=etaInv *np.dot(K, yh)
    predicted_label = np.sign(f - rho)
    predicted_label = np.sign(f)

    EVAL_Validation=Evaluate(Y, predicted_label, 1)
    end = time.time()
    Time=end - start
    return EVAL_Validation,Time