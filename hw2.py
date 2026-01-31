import numpy as np
import pandas as pd
import pickle


def loadData(fis):
    with open(fis, 'rb') as f:
        date = pickle.load(f)
    return date



def prepareData(date):
    imagini, etichete = zip(*date)
    X = np.array([i.flatten() for i in imagini], dtype=np.float32) / 255.0  
    Y = np.array(etichete)
    return X, Y


def softmax(Z):
    Z1 = Z - np.max(Z, axis=1, keepdims=True)
    exp = np.exp(Z1)
    probabilitati = exp / np.sum(exp, axis=1, keepdims=True)
    
    return probabilitati

def oneHotEncoding(Y, clase):
    batchSize = Y.shape[0]
    one_hot = np.zeros((batchSize, clase), dtype=np.float32)
    one_hot[np.arange(batchSize), Y] = 1
    
    return one_hot

def loss(P, Y):
    batchSize = P.shape[0]
    epsilon = 1e-15
    clipp = np.clip(P, epsilon, 1.0 - epsilon)
    
    crossEntropy = -np.sum(Y * np.log(clipp)) / batchSize

    return crossEntropy



def acuratete(P, Y):
    predictii = np.argmax(P, axis=1)
    return np.mean(predictii == Y)



def initializare(inputSize, outputSize):
    limita = np.sqrt(6.0 / (inputSize + outputSize))
    weight = np.random.uniform(-limita, limita, size=(inputSize, outputSize)).astype(np.float32)
    bias = np.zeros((1, outputSize), dtype=np.float32)
    return weight, bias



def forwardPropagation(X, weight, bias):
    scor = X @ weight + bias
    probabilitati = softmax(scor)
    return scor, probabilitati



def backwardPropagation(X, Y_oneHot, probabilitati):
    batchSize = X.shape[0]
    dScor = probabilitati - Y_oneHot
    dWeight = (X.T @ dScor) / batchSize
    dBias = np.sum(dScor, axis=0, keepdims=True) / batchSize
    return dWeight, dBias



def update(weight, bias, dWeight, dBias, lr):
    weight = weight - lr * dWeight
    bias = bias - lr * dBias
    return weight, bias



def train(xTrain, yTrain, xTest, yTest, epochs, lr, batchSize):
    m_train, inputSize = xTrain.shape
    clase = len(np.unique(yTrain))
    weight, bias = initializare(inputSize, clase)

    for epoch in range(epochs):
        permutation = np.random.permutation(m_train)
        xShuffled = xTrain[permutation]
        yShuffled = yTrain[permutation]
        epoch_loss = 0
        num_batches = 0
        for i in range(0, m_train, batchSize):
            xBatch = xShuffled[i : i + batchSize]
            yBatch = yShuffled[i : i + batchSize]
            y_oneHot = oneHotEncoding(yBatch, clase)
            _, probabilitati = forwardPropagation(xBatch, weight, bias)
            
            batch_loss = loss(probabilitati, y_oneHot)
            epoch_loss += batch_loss
            num_batches += 1
            dWeight, dBias = backwardPropagation(xBatch, y_oneHot, probabilitati)
            weight, bias = update(weight, bias, dWeight, dBias, lr)
            
        _, P = forwardPropagation(xTest, weight, bias)
        acc = acuratete(P, yTest)
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.5f} - Acuratete: {acc:.5f}")

    return weight, bias


def submission(weight, bias, xTest):
    _, probabilitati = forwardPropagation(xTest, weight, bias)
    predictii = np.argmax(probabilitati, axis=1)
    df = pd.DataFrame({
        "ID": range(len(predictii)),
        "target": predictii
    })
    df.to_csv("submission.csv", index=False)



if __name__ == "__main__":
    EPOCHS = 200
    LR = 0.064
    BATCHSIZE = 64

    train_data = loadData("./fii-nn-2025-homework-2/extended_mnist_train.pkl")
    test_data = loadData("./fii-nn-2025-homework-2/extended_mnist_test.pkl")
    xTrain, yTrain = prepareData(train_data)
    xTest, yTest = prepareData(test_data)
    weight, bias = train(xTrain, yTrain, xTest, yTest, EPOCHS, LR, BATCHSIZE)

    _, probabilitati = forwardPropagation(xTest, weight, bias)

    acc = acuratete(probabilitati, yTest)
    print(f"\nAcuratețe finală pe test: {acc:.5f}%")
    submission(weight, bias, xTest)