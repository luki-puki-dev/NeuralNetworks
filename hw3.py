import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn.functional as F


def loadData(fis):
    with open(fis, 'rb') as f:
        date = pickle.load(f)
    return date


def prepareData(date):
    imagini, etichete = zip(*date)
    X = np.array([i.flatten() for i in imagini], dtype=np.float32) / 255.0  
    Y = np.array(etichete)
    return X, Y


def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return (Z > 0).astype(np.float32)


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


def loss_torch(P, Y):
    P_tensor = torch.from_numpy(P)
    Y_tensor = torch.from_numpy(Y).long()
    loss_value = F.cross_entropy(P_tensor, Y_tensor)
    return loss_value.item()


def acuratete(P, Y):
    predictii = np.argmax(P, axis=1)
    return np.mean(predictii == Y)


def initializare(inputSize, hiddenSize, outputSize):
    limita1 = np.sqrt(6.0 / (inputSize + hiddenSize))
    W1 = np.random.uniform(-limita1, limita1, size=(inputSize, hiddenSize)).astype(np.float32)
    b1 = np.zeros((1, hiddenSize), dtype=np.float32)
    
    limita2 = np.sqrt(6.0 / (hiddenSize + outputSize))
    W2 = np.random.uniform(-limita2, limita2, size=(hiddenSize, outputSize)).astype(np.float32)
    b2 = np.zeros((1, outputSize), dtype=np.float32)
    
    return W1, b1, W2, b2


def dropout_mask(shape, dropout_rate):
    if dropout_rate == 0:
        return np.ones(shape, dtype=np.float32)
    mask = (np.random.rand(*shape) > dropout_rate).astype(np.float32)
    mask /= (1.0 - dropout_rate)
    return mask


def forwardPropagation(X, W1, b1, W2, b2, dropout_rate=0.0, training=True):
    Z1 = X @ W1 + b1  
    A1 = relu(Z1)      

    dropout_m = None
    if training and dropout_rate > 0:
        dropout_m = dropout_mask(A1.shape, dropout_rate)
        A1 = A1 * dropout_m
    Z2 = A1 @ W2 + b2  
    P = softmax(Z2)   
    
    return Z1, A1, Z2, P, dropout_m


def backwardPropagation(X, Y_oneHot, Z1, A1, P, W2, dropout_m):
    batchSize = X.shape[0]
    dZ2 = P - Y_oneHot   
    dW2 = (A1.T @ dZ2) / batchSize 
    db2 = np.sum(dZ2, axis=0, keepdims=True) / batchSize 
    
    dA1 = dZ2 @ W2.T
    
    if dropout_m is not None:
        dA1 = dA1 * dropout_m
    
    dZ1 = dA1 * relu_derivative(Z1)  
    dW1 = (X.T @ dZ1) / batchSize  
    db1 = np.sum(dZ1, axis=0, keepdims=True) / batchSize
    
    return dW1, db1, dW2, db2


def update(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2
    return W1, b1, W2, b2


def evaluate(X, Y, W1, b1, W2, b2):
    _, _, Z2, P, _ = forwardPropagation(X, W1, b1, W2, b2, dropout_rate=0.0, training=False)
    loss_val = loss_torch(Z2, Y)
    acc = acuratete(P, Y)
    return loss_val, acc


def train(xTrain, yTrain, xTest, yTest, epochs, lr, batchSize, hiddenSize=100, dropout_rate=0.2):
    m_train, inputSize = xTrain.shape
    clase = len(np.unique(yTrain))
    W1, b1, W2, b2 = initializare(inputSize, hiddenSize, clase)

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
            
            Z1, A1, Z2, P, dropout_m = forwardPropagation(
                xBatch, W1, b1, W2, b2, dropout_rate=dropout_rate, training=True
            )
            
            batch_loss = loss_torch(Z2, yBatch)
            epoch_loss += batch_loss
            num_batches += 1
            
            dW1, db1, dW2, db2 = backwardPropagation(xBatch, y_oneHot, Z1, A1, P, W2, dropout_m)
            W1, b1, W2, b2 = update(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)
        
        avg_train_loss = epoch_loss / num_batches
        train_loss, train_acc = evaluate(xTrain, yTrain, W1, b1, W2, b2)
        val_loss, val_acc = evaluate(xTest, yTest, W1, b1, W2, b2)
        
        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.5f} - "
              f"Val Loss: {val_loss:.5f}, Val Acc: {val_acc:.5f}")

    return W1, b1, W2, b2


def submission(W1, b1, W2, b2, xTest):
    _, _, _, probabilitati, _ = forwardPropagation(
        xTest, W1, b1, W2, b2, dropout_rate=0.0, training=False
    )
    predictii = np.argmax(probabilitati, axis=1)
    df = pd.DataFrame({
        "ID": range(len(predictii)),
        "target": predictii
    })
    df.to_csv("submission.csv", index=False)



if __name__ == "__main__":
    EPOCHS = 150
    LR = 0.08
    BATCHSIZE = 50
    HIDDEN_SIZE = 100
    DROPOUT_RATE = 0.15

    train_data = loadData("./fii-nn-2025-homework-2/extended_mnist_train.pkl")
    test_data = loadData("./fii-nn-2025-homework-2/extended_mnist_test.pkl")
    xTrain, yTrain = prepareData(train_data)
    xTest, yTest = prepareData(test_data)
    
    W1, b1, W2, b2 = train(
        xTrain, yTrain, xTest, yTest, 
        EPOCHS, LR, BATCHSIZE, 
        hiddenSize=HIDDEN_SIZE, 
        dropout_rate=DROPOUT_RATE
    )

    _, _, _, probabilitati, _ = forwardPropagation(
        xTest, W1, b1, W2, b2, dropout_rate=0.0, training=False
    )
    acc = acuratete(probabilitati, yTest)
    print(f"\nacuratete finala: {acc:.5f}")
    
   
    submission(W1, b1, W2, b2, xTest)