
def extrageCoeficient(termenStr):

    if 'x' in termenStr:
        coefStr = termenStr.replace('x', '')
    elif 'y' in termenStr:
        coefStr = termenStr.replace('y', '')
    elif 'z' in termenStr:
        coefStr = termenStr.replace('z', '')
    else:
        return 0 

    if coefStr == '':
        return 1  
    elif coefStr == '-':
        return -1 
    else:
        return int(coefStr) 

def parseazaSistem(numeFisier):
    matriceA = []
    vectorB = []
    
    try:
        with open(numeFisier, 'r') as f:
            for linie in f:
                linie = linie.strip()
                if not linie:
                    continue
                
                parti = linie.split('=')
                lhs = parti[0]
                rhs = parti[1]
                vectorB.append(int(rhs.strip()))
                lhs = lhs.replace('-', '+-')
                termeni = [t for t in lhs.split('+') if t]
                
                randA = [0, 0, 0]
                
                for termen in termeni:
                    valoare = extrageCoeficient(termen)
                    if 'x' in termen:
                        randA[0] = valoare
                    elif 'y' in termen:
                        randA[1] = valoare
                    elif 'z' in termen:
                        randA[2] = valoare
                        
                matriceA.append(randA)
                
    except FileNotFoundError:
        print(f"Eroare: Fisierul '{numeFisier}' nu a fost gasit.")
        return None, None
        
    return matriceA, vectorB


def calculeazaDeterminant(matrice):
    a11 = matrice[0][0]
    a12 = matrice[0][1]
    a13 = matrice[0][2]
    
    a21 = matrice[1][0]
    a22 = matrice[1][1]
    a23 = matrice[1][2]
    
    a31 = matrice[2][0]
    a32 = matrice[2][1]
    a33 = matrice[2][2]
    
    det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)
    return det

def calculeazaUrma(matrice):
    urma = matrice[0][0] + matrice[1][1] + matrice[2][2]
    return urma

def calculeazaNormaVector(vector):
    sumaPatrate = vector[0]**2 + vector[1]**2 + vector[2]**2
    norma = sumaPatrate**0.5
    return norma

def calculeazaTranspusa(matrice):
    transpusa = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            transpusa[j][i] = matrice[i][j]
            
    return transpusa

def inmultireMatriceVector(matrice, vector):
    rezultat = [0, 0, 0]
    
    for i in range(3): 
        sumaRand = 0
        for j in range(3): 
            sumaRand += matrice[i][j] * vector[j]
        rezultat[i] = sumaRand
        
    return rezultat


def obtineMatricePentruCramer(matriceA, vectorB, coloanaIndex):
    
    copieA = [rand[:] for rand in matriceA]
    
    for i in range(3):
        copieA[i][coloanaIndex] = vectorB[i]
        
    return copieA

def rezolvaPrinCramer(matriceA, vectorB):
    detA = calculeazaDeterminant(matriceA)
    
    if detA == 0:
        print("nu se poate prin Cramemr")
        return None
        
    matriceAx = obtineMatricePentruCramer(matriceA, vectorB, 0)
    detAx = calculeazaDeterminant(matriceAx)

    matriceAy = obtineMatricePentruCramer(matriceA, vectorB, 1)
    detAy = calculeazaDeterminant(matriceAy)
    

    matriceAz = obtineMatricePentruCramer(matriceA, vectorB, 2)
    detAz = calculeazaDeterminant(matriceAz)
    
    x = detAx / detA
    y = detAy / detA
    z = detAz / detA
    
    return [x, y, z]


def determinant2x2(mat):
    return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

def obtineMinor(matrice, rand, col):
    minor = []
    for i in range(3):
        if i == rand:
            continue
        randNou = []
        for j in range(3):
            if j == col:
                continue
            randNou.append(matrice[i][j])
        minor.append(randNou)
    return minor

def calculeazaMatriceCofactori(matrice):
    cofactori = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    for i in range(3):
        for j in range(3):
            minor = obtineMinor(matrice, i, j)
            detMinor = determinant2x2(minor)
            semn = 1 if (i + j) % 2 == 0 else -1
            cofactori[i][j] = semn * detMinor
            
    return cofactori

def calculeazaInversa(matriceA):
    detA = calculeazaDeterminant(matriceA)
    
    if detA == 0:
        print("matricea nu e inversabila")
        return None
        

    cofactori = calculeazaMatriceCofactori(matriceA)
    adjuncta = calculeazaTranspusa(cofactori)
    
    inversa = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    factor = 1.0 / detA
    
    for i in range(3):
        for j in range(3):
            inversa[i][j] = adjuncta[i][j] * factor
            
    return inversa

def rezolvaPrinInversare(matriceA, vectorB):
    inversaA = calculeazaInversa(matriceA)
    
    if inversaA is None:
        return None
    
    solutieX = inmultireMatriceVector(inversaA, vectorB)
    return solutieX

def afiseazaMatrice(matrice, nume):
    print(f"\nMatricea {nume}:")
    if matrice:
        for rand in matrice:
            print("  [" + ", ".join(f"{elem:8.4f}" for elem in rand) + "]")
    else:
        print("  None")

def afiseazaVector(vector, nume):
    print(f"\nVectorul {nume}:")
    if vector:
        print("  [" + ", ".join(f"{elem:8.4f}" for elem in vector) + "]")
    else:
        print("  None")


def main():

    matriceA, vectorB = parseazaSistem('sistem.txt')
    
    if matriceA is None:
        return 

    afiseazaMatrice(matriceA, "A")
    afiseazaVector(vectorB, "B")
    
    detA = calculeazaDeterminant(matriceA)
    print(f"\n Det  A: {detA}")
    
    urmaA = calculeazaUrma(matriceA)
    print(f"\n Trace A: {urmaA}")
    
    normaB = calculeazaNormaVector(vectorB)
    print(f"\n Norma Euclidiana  B: {normaB:.4f}")
    
    transpusaA = calculeazaTranspusa(matriceA)
    afiseazaMatrice(transpusaA, "A Transpusa")
    
    produsAB = inmultireMatriceVector(matriceA, vectorB)
    afiseazaVector(produsAB, "A * B ca operatie")
    
  
    
    solutieCramer = rezolvaPrinCramer(matriceA, vectorB)
    afiseazaVector(solutieCramer, "Solutie X (Cramer)")
    

    
    solutieInversare = rezolvaPrinInversare(matriceA, vectorB)
    afiseazaVector(solutieInversare, "Solutie X (Inversare)")
    
    if solutieInversare:
        verificare = inmultireMatriceVector(matriceA, solutieInversare)
        afiseazaVector(verificare, "A * X")
        print("(Ar trebui sa fie egal cu vectorul B)")
        

if __name__ == "__main__":
    main()