import os
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import time

# --- Dataset ---
class ExtendedMNISTDataset(Dataset):
    def __init__(self, root: str = "/kaggle/input/fii-nn-2025-homework-4", train: bool = True):
        file = "extended_mnist_test.pkl"
        if train:
            file = "extended_mnist_train.pkl"
        file = os.path.join(root, file)
        with open(file, "rb") as fp:
            self.data = pickle.load(fp)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int):
        return self.data[i]

# --- Activations ---
def relu(z):
    return np.maximum(0, z)

def reluDerivative(z):
    return (z > 0).astype(np.float32)

def softmax(z):
    z_max = np.max(z, axis=1, keepdims=True)
    exp = np.exp(z - z_max)
    return exp / np.sum(exp, axis=1, keepdims=True)

def oneHotEncoding(y, classes, labelSmoothing=0.0):
    m = y.shape[0]
    oneHot = np.full((m, classes), labelSmoothing / (classes - 1), dtype=np.float32)
    oneHot[np.arange(m), y] = 1.0 - labelSmoothing
    return oneHot

# --- Augmentation (Optimized) ---
def fastAugmentBatch(xBatch, shift=2):
    b, inputDim = xBatch.shape
    size = int(np.sqrt(inputDim))
    
    # Reshape și Pad
    xImg = xBatch.reshape(b, size, size)
    xPad = np.pad(xImg, ((0,0), (shift, shift), (shift, shift)), mode='constant')
    
    # Random crops vectorizat
    cropX = np.random.randint(0, 2*shift + 1, size=b)
    cropY = np.random.randint(0, 2*shift + 1, size=b)
    
    # Pre-calculăm grid-ul de bază o singură dată (broadcasting implicit)
    base_idx = np.arange(size)
    rows = base_idx[None, :, None] + cropY[:, None, None]
    cols = base_idx[None, None, :] + cropX[:, None, None]
    
    # Advanced indexing
    batchIdx = np.arange(b)[:, None, None]
    return xPad[batchIdx, rows, cols].reshape(b, -1)

# --- Initialization ---
def initializareHe(inputSize, outputSize):
    std = np.sqrt(2.0 / inputSize)
    weights = np.random.randn(inputSize, outputSize).astype(np.float32) * std
    bias = np.zeros((1, outputSize), dtype=np.float32)
    return weights, bias

def initNetwork(inputSize, hidden1, hidden2, outputSize):
    w1, b1 = initializareHe(inputSize, hidden1)
    w2, b2 = initializareHe(hidden1, hidden2)
    w3, b3 = initializareHe(hidden2, outputSize)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3}

def dropoutMask(shape, dropoutRate):
    if dropoutRate <= 0: return 1.0
    mask = (np.random.rand(*shape) > dropoutRate).astype(np.float32)
    # In-place multiplication is faster later
    mask *= (1.0 / (1.0 - dropoutRate)) 
    return mask

# --- Forward / Backward ---
def forwardPropagation(x, params, dropoutRate=0.0, training=True):
    w1, b1 = params["w1"], params["b1"]
    w2, b2 = params["w2"], params["b2"]
    w3, b3 = params["w3"], params["b3"]
    
    z1 = x @ w1 + b1
    a1 = relu(z1)
    mask1 = None
    if training and dropoutRate > 0:
        mask1 = dropoutMask(a1.shape, dropoutRate)
        a1 *= mask1
        
    z2 = a1 @ w2 + b2
    a2 = relu(z2)
    mask2 = None
    if training and dropoutRate > 0:
        mask2 = dropoutMask(a2.shape, dropoutRate)
        a2 *= mask2
        
    z3 = a2 @ w3 + b3
    p = softmax(z3)
    
    return p, (z1, a1, mask1, z2, a2, mask2, z3, p)

def backwardPropagation(x, yOneHot, params, cache):
    (z1, a1, mask1, z2, a2, mask2, z3, p) = cache
    w2, w3 = params["w2"], params["w3"]
    batchSize = x.shape[0]
    invBatch = 1.0 / batchSize # Optimizare: împărțim o singură dată

    dz3 = p - yOneHot  
    dw3 = (a2.T @ dz3) * invBatch 
    db3 = np.sum(dz3, axis=0, keepdims=True) * invBatch
    
    da2 = dz3 @ w3.T 
    if mask2 is not None: da2 *= mask2
    dz2 = da2 * reluDerivative(z2)
    dw2 = (a1.T @ dz2) * invBatch
    db2 = np.sum(dz2, axis=0, keepdims=True) * invBatch
    
    da1 = dz2 @ w2.T
    if mask1 is not None: da1 *= mask1
    dz1 = da1 * reluDerivative(z1)
    dw1 = (x.T @ dz1) * invBatch
    db1 = np.sum(dz1, axis=0, keepdims=True) * invBatch
    
    return {"dw1": dw1, "db1": db1, "dw2": dw2, "db2": db2, "dw3": dw3, "db3": db3}

# --- Adam Optimizer (In-Place Optimization) ---
def initAdam(params):
    v, s = {}, {}
    for key in params:
        v[key] = np.zeros_like(params[key])
        s[key] = np.zeros_like(params[key])
    return v, s

def updateAdam(params, grads, v, s, t, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weightDecay=1e-4):
    # Pre-calculăm scalarii o singură dată pe pas, nu per parametru
    biasCorr1 = 1.0 - (beta1 ** t)
    biasCorr2 = 1.0 - (beta2 ** t)
    
    # Folosim operații In-Place (+=, *=) pentru viteză și memorie
    for key in params:
        grad = grads["d" + key]
        
        # Weight Decay
        if 'w' in key:
            grad += weightDecay * params[key]

        # Momentum (v) anulează "zgomotul" specific fiecărui batch mic și păstrează direcția care reduce eroarea pentru toate tipurile de 7 din setul de date, nu doar pentru cele din batch-ul curent.
        v[key] *= beta1                           
        v[key] += (1 - beta1) * grad
        
        # RMSProp (s) -> s = b2*s + (1-b2)*g^2
        s[key] *= beta2
        s[key] += (1 - beta2) * (grad ** 2)
        
        # Bias Correction & Update
        # Calculăm update-ul
        v_hat = v[key] / biasCorr1
        s_hat = s[key] / biasCorr2
        
        update_step = v_hat
        update_step /= (np.sqrt(s_hat) + epsilon)
        update_step *= lr
        
        params[key] -= update_step
        
    return params, v, s

# --- Train Loop ---
def train(xTrain, yTrainIndices, xTest, epochs, lr, batchSize, hidden1, hidden2, dropoutRate, numClasses):
    mTrain, inputSize = xTrain.shape
    
    # One Hot pre-calculat
    yTrainEncoded = oneHotEncoding(yTrainIndices, numClasses, labelSmoothing=0.1)
    
    params = initNetwork(inputSize, hidden1, hidden2, numClasses)
    v, s = initAdam(params)
    t = 0
    
    bestParams = params.copy()

    print(f"Start Training: {mTrain} samples, {epochs} epochs.")
    start_total = time.time()

    for epoch in range(epochs):
        permutation = np.random.permutation(mTrain)
        
        # LR Schedule
        if epoch < int(epochs * 0.5): currentLr = lr
        elif epoch < int(epochs * 0.8): currentLr = lr * 0.2
        else: currentLr = lr * 0.05
        
        # Batch Loop
        for i in range(0, mTrain, batchSize):
            t += 1
            indices = permutation[i : i + batchSize]
            
            # Slicing rapid (fără copiere masivă)
            xBatch = xTrain[indices]
            yBatchEncoded = yTrainEncoded[indices]
            
            # Augment
            xBatchAug = fastAugmentBatch(xBatch, shift=2)
            
            # Forward / Back / Update
            p, cache = forwardPropagation(xBatchAug, params, dropoutRate, training=True)
            grads = backwardPropagation(xBatchAug, yBatchEncoded, params, cache)
            params, v, s = updateAdam(params, grads, v, s, t, lr=currentLr, weightDecay=1e-4)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} done.")
            bestParams = {k: val.copy() for k, val in params.items()}

    print(f"Training finished in {time.time() - start_total:.1f}s")
    return bestParams

if __name__ == "__main__":
    # --- FAST DATA LOADING (Vectorized) ---
    print("Loading data...")
    # Încărcăm tot setul dintr-o dată, fără loop-uri Python lente
    datasetTrain = ExtendedMNISTDataset(train=True)
    images_train, labels_train = zip(*datasetTrain.data)
    
    # Conversie directă și reshape
    xTrain = np.array(images_train, dtype=np.float32).reshape(len(images_train), -1) / 255.0
    yTrainRaw = np.array(labels_train) # Păstrăm tipul original momentan

    # Mapare clase
    uniqueClasses = np.unique(yTrainRaw)
    classMap = {label: idx for idx, label in enumerate(uniqueClasses)}
    inverseClassMap = {idx: label for label, idx in classMap.items()}
    yTrainIndices = np.array([classMap[label] for label in yTrainRaw], dtype=np.int32)
    
    # Test Data Load
    datasetTest = ExtendedMNISTDataset(train=False)
    # Testul are doar imagini (tuple de 1 element sau direct imaginea? depinde de pickle)
    # Presupunem formatul standard [(img, label), ...] sau [(img), ...]
    # Verificăm structura rapid:
    test_data_unpacked = zip(*datasetTest.data)
    # ExtendedMNISTDataset returneaza (img, label) chiar si la test?
    # De obicei test set pe Kaggle are (img, id) sau doar img.
    # Ne bazăm pe codul tău anterior: `image, label in dataset`.
    images_test, _ = test_data_unpacked # Ignorăm label-ul fictiv de test dacă există
    xTest = np.array(images_test, dtype=np.float32).reshape(len(images_test), -1) / 255.0

    print("Data loaded & processed.")

    # Hiperparametri
    EPOCHS = 45
    LR = 0.001
    BATCHSIZE = 256
    HIDDEN1 = 800
    HIDDEN2 = 400
    DROPOUT = 0.25
    NUMCLASSES = len(uniqueClasses)

    # Train
    bestParams = train(xTrain, yTrainIndices, xTest, EPOCHS, LR, BATCHSIZE, HIDDEN1, HIDDEN2, DROPOUT, NUMCLASSES)

    # Submission
    pFinal, _ = forwardPropagation(xTest, bestParams, dropoutRate=0.0, training=False)
    predictionsIndices = np.argmax(pFinal, axis=1)
    
    # Mapare inversă (Index -> Label Real)
    predictionsReal = [inverseClassMap[idx] for idx in predictionsIndices]

    predictionsCsv = {
        "ID": range(len(predictionsReal)),
        "target": predictionsReal,
    }
    
    df = pd.DataFrame(predictionsCsv)
    df.to_csv("submission.csv", index=False)
    print("Submission saved successfully.")