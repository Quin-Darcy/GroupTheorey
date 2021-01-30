#!/usr/bin/env python3
import numpy as np
from PIL import Image
import itertools
from itertools import permutations
import time
import math
import decimal
#-------------------------------------------------------------
#---------------------------Misc------------------------------
#-------------------------------------------------------------
def mean(ang):
    N = len(ang)
    k = 0
    for i in range(N):
        k += ang[i]
    return k/N
def SD(ang):
    N = len(ang)
    mu = mean(ang)
    k = 0
    for i in range(N):
        k += (ang[i]-mu)**2
    return math.sqrt(k/N)
#-------------------------------------------------------------
#--------------------------Colors-----------------------------
#-------------------------------------------------------------
def mult(x,y,z):
    arr = [[math.sqrt(3)/2, 0, 1/2], [-math.sqrt(2)/4, math.sqrt(2)/2,
        math.sqrt(6)/4], [-math.sqrt(2)/4, -math.sqrt(2    )/2, math.sqrt(6)/4]]
    vec = [x,y,z]
    newv = [0,0,0]
    for i in range(3):
        for j in range(3):
            newv[i] += int(vec[j]*arr[i][j])
    return newv

def setColor(t):
    arr = [0,0,0]
    k = (math.pi)/3
    a = math.sqrt(3)
    theta = (math.pi/2)-np.arccos(a/3)
    r = (127.5)*math.sqrt(3)
    c = r*a
    if (0<= t < k):
        x = c/(math.tan(t)+a)
        y = math.tan(t)*x
        z = (-(1/r)*2*x+2)*255*(math.sqrt(2)*math.sin(theta)-1/2)+127.5
        arr = mult(x,y,z)
    if (k<= t < 2*k):
        x = c/(2*math.tan(t))
        y = math.tan(t)*x
        z = -(1/r)*(x-r/2)*255*(1/2-math.sqrt(2)*math.sin(theta))+255*math.sqrt(2)*math.sin(theta)
        arr = mult(x,y,z)
    if (2*k<= t < 3*k):
        x = c/(math.tan(t)-a)
        y = math.tan(t)*x
        z = -(1/r)*(2*x+r)*255*(math.sqrt(2)*math.sin(theta)-1/2)+127.5
        arr = mult(x,y,z)
    if (3*k<= t < 4*k):
        x = -c/(math.tan(t)+a)
        y = math.tan(t)*x
        z = (1/r)*(2*x+2*r)*255*(1/2-math.sqrt(2)*math.sin(theta))+255*math.sqrt(2)*math.sin(theta)
        arr = mult(x,y,z)
    if (4*k<= t < 5*k):
        x = -c/(2*math.tan(t))
        y = math.tan(t)*x
        z = (1/r)*(x+r/2)*255*(math.sqrt(2)*math.sin(theta)-1/2)+127.5
        arr = mult(x,y,z)
    if (5*k <= t < 6*k):
        x = -c/(math.tan(t)-a)
        y = math.tan(t)*x
        z = (1/r)*(2*x-r)*255*(1/2-math.sqrt(2)*math.sin(theta))+255*math.sqrt(2)*math.sin(theta)
        arr = mult(x,y,z)
    return arr
def get_Colors(angles):
    L = len(angles)
    Colors = [[0,0,0] for i in range(L)]
    for i in range(L):
        Colors[i] = setColor(angles[i])
        for j in range(3):
            if (Colors[i][j] < 0):
                Colors[i][j] = 0
    return Colors
#-------------------------------------------------------------
#------------------------Permutations-------------------------
#-------------------------------------------------------------
def swap(arr,  n):
    k = 0
    arr1 = []
    for i in range(n-1):
        temp = arr[i]
        arr[i] = arr[i+1]
        arr[i+1] = temp
        arr1.append(arr.copy())
    arr = list(arr1)
    return arr

def get_Sn(arr, n):
    L = 0
    c = 0
    p = swap(arr, n)
    fact = math.factorial(n)
    while (L < fact):
        k = len(p)
        for i in range(c, k+1):
            temp = swap(p[i], n)
            m = len(temp)
            for j in range(m):
                p.append(temp[j])
            c += 1
        p.sort()
        p = list(p for p,_ in itertools.groupby(p))
        L = len(p)
    return p
#------------------------------------------------------------
#----------------------Multiplication------------------------
#------------------------------------------------------------
def multiply(a, b):
    n = len(a)
    product = [0]*n
    for i in range(n):
        for j in range(n):
            if (a[j] == i+1):
                product[j] = b[i]
    return product
#------------------------------------------------------------
#-------------------Coordinates/Angles-----------------------
#------------------------------------------------------------
def d(V, U):
    decimal.getcontext().prec = 40
    n = len(V)
    SUM = decimal.Decimal(0)
    for i in range(n):
        SUM += (V[i]-U[i])**2
    SUM = decimal.Decimal(SUM).sqrt()
    return SUM
def innerP(U, V):
    decimal.getcontext().prec = 40 
    n = len(U)
    ip = 0
    for i in range(n):
        ip += decimal.Decimal(U[i]*V[i])
    return ip
def get_Angles(coors, n):
    decimal.getcontext().prec = 40 
    L = len(coors)
    origin = [0]*n
    d1 = decimal.Decimal(d(coors[0], origin))
    angles = [0]*L
    Y = decimal.Decimal(2*math.pi)
    for i in range(L):
        iP = decimal.Decimal(innerP(coors[0], coors[i]))
        d2 = decimal.Decimal(d(coors[i], origin))
        if (iP/(d1*d2) > 1):
            diff = decimal.Decimal(iP/(d1*d2))-1
            new = decimal.Decimal(iP/(d1*d2))-diff
            X = decimal.Decimal(math.acos(new))
            angles[i] = X%Y
        elif (iP/(d1*d2) < -1):
            diff = 1 - decimal.Decimal(iP/(d1*d2))
            new = decimal.Decimal(iP/(d1*d2))+diff
            X = decimal.Decimal(math.acos(new))
            angles[i] = X%Y
        else:
            X = decimal.Decimal(iP/(d1*d2))
            X = decimal.Decimal(math.acos(X))
            angles[i] = X%Y
    for i in range(L):
        for j in range(i, L):
            if (angles[i] == angles[j]):
                dist = d(coors[i], coors[j])
                if (dist != 0):
                    angles[j] += decimal.Decimal(1/dist)
                    angles[j] = angles[j]%Y
                else:
                    W = decimal.Decimal(math.pi)
                    angles[j] += W
                    angles[j] = angles[j]%Y
    return angles
def get_Coors(Sn, n):
    L = len(Sn)
    coors = [[0]*n for i in range(L)]
    for i in range(L):
        for j in range(n):
            k = abs(Sn[i][j]-j)
            coors[i][j] = ((-1)**(k))*(get_Prime(Sn[i][j])+k)**(j+1)
    return coors
#------------------------------------------------------------
#--------------------------FAIL------------------------------
#------------------------------------------------------------
def failAngles(Sn, n):
    L = len(Sn)
    a = 0
    theta = decimal.Decimal(2*math.pi/L)
    angles = [0]*L
    for i in range(L):
        angles[i] = a
        a += theta
    return angles
#------------------------------------------------------------
#-------------------------Tables-----------------------------
#------------------------------------------------------------
def get_Prime(n):
    val = False
    p = 2
    count = 1
    num = 0
    PrimeCount = 0
    while (val != True):
        for i in range(1, p+1):
            if (p%i == 0):
                num += 1
        if (num > 2):
            p += 1
        else:
            PrimeCount += 1
            p += 1
        if (PrimeCount == n):
            val = True
        else:
            val = False
        num = 0
        count += 1
    return p-1

def get_Table(Sn, n):
    L = len(Sn)
    table = [[[0]*n]*L for i in range(L)]
    for i in range(L):
        for j in range(L):
            table[i][j] = multiply(Sn[i], Sn[j])
    return table
#-----------------------------------------------------------
#-----------------------CayleyPic---------------------------
#-----------------------------------------------------------
def MakePicture(Colors, cTable, Sn):
    c1 = 0
    c2 = 0
    k = 1
    n = len(Sn)
    L = int(10*n**2)
    M = 10*n
    L2 = len(Sn)
    im = Image.new('RGB', (L, L), color='white')
    im2 = Image.new('RGB', (L+M, L+M), color='black')
    px2 = im2.load()
    px = im.load()
    for y in range(L):
        if (y%(M) == 0):
            if (y != 0):
                c2 += 1
        for x in range(L):
            if (x%(M) == 0):
                if (x != 0):
                    c1 += 1
            perm = cTable[c2][c1]
            for i in range(L2):
                if (perm == Sn[i]):
                    px[x,y] = tuple(Colors[i])
        c1 = 0
    for y in range(M):
        for x in range(M, L+M):
            px2[x,y] = px[x-M, y]
    for y in range(M, L+M):
        for x in range(M):
            px2[x, y] = px[x, y-M]
    for y in range(M, L+M):
        for x in range(M, L+M):
            px2[x,y] = px[x-M, y-M]
    for i in range(L+M):
        if (i%(M) == 0):
            for j in range(L+M):
                px2[i, j] = (0,0,0)
                px2[j, i] = (0,0,0)
    im2.save('test.png')
    return [px2, L+M, M]
#-----------------------------------------------------------
#-------------------------Subgroups-------------------------
#-----------------------------------------------------------
def fMult(A):
    L = len(A)
    lst = []*(L**2)
    for i in range(L):
        for j in range(L):
            lst.append(multiply(A[i], A[j]).copy())
    lst = [x for x in lst if x != []]
    lst.sort()
    lst = list(lst for lst,_ in itertools.groupby(lst))
    return lst
def Cyclics(Sn):
    k = 0
    L = len(Sn)
    n = len(Sn[0])
    cyclics = [[] for i in range(L)]
    iden = Sn[0]
    temp = Sn.copy()
    for i in range(L):
        if (temp[i] != [0,0]):
            A = Sn[i].copy()
            cyclics[k].append(A.copy())
            temp = [[0,0] if x == A else x for x in temp]
            while (A != iden):
                A = multiply(A, Sn[i])
                cyclics[k].append(A.copy())
                temp = [[0,0] if x == A else x for x in temp]
                cyclics[k].sort()
            k += 1
    cyclics = [x for x in cyclics if x != []]
    cyclics.sort()
    cyclics = list(cyclics for cyclics,_ in itertools.groupby(cyclics))
    L = len(cyclics)
    for i in range(L):
        print(i+1, len(cyclics[i]))
    subNum = int(input('Enter number between 1 and '+str(L)+': '))
    print(Sn[subNum-1])
    return [cyclics, subNum]
def Subs(Sn):   # Need to change so more than just cyclic subs are found
    cyclics = Cyclics(Sn)
    return [cyclics, subNum]
#-----------------------------------------------------------
#-------------------------Stabilizer------------------------
#-----------------------------------------------------------
def Inverse(g, cTable):
    idx = 0
    inv = 0
    iden = cTable[0][0]
    inverse = []
    L = len(cTable[0])
    for i in range(L):
        if (g == cTable[i][0]):
            idx = i
            break
    for i in range(L):
        if (cTable[idx][i] == iden):
            inv = i
            break
    return inv
def Stabilizer(cTable):
    k = 0
    element = int(input('Choose number between 1 and '+str(len(cTable[0]))+': '))
    s = cTable[0][element-1]
    L = len(cTable[0])
    n = len(cTable[0][0])
    stab = [[]*n for i in range(L)]
    for i in range(L):
        g = cTable[i][0]
        gi = cTable[0][Inverse(g, cTable)]
        if (multiply(multiply(g,s), gi) == s):
            stab[k] = g
            k += 1
    stab = [x for x in stab if x != []]
    stab.sort()
    stab = list(stab for stab,_ in itertools.groupby(stab))
    print('s = ', s)
    return stab
#-----------------------------------------------------------
#----------------------------Orbit--------------------------
#-----------------------------------------------------------
def Orbit(cTable):
    k = 0
    element = int(input('Choose number between 1 and '+str(len(cTable[0]))+': '))
    s = cTable[0][element-1]
    L = len(cTable[0])
    n = len(cTable[0][0])
    orb = [[]*n for i in range(L)]
    for i in range(L):
        g = cTable[i][0]
        gi = cTable[0][Inverse(g, cTable)]
        orb[k] = multiply(multiply(g,s), gi)
        k += 1
    orb = [x for x in orb if x!= []]
    orb.sort()
    orb = list(orb for orb,_ in itertools.groupby(orb))
    print('s = ', s)
    return orb
#-----------------------------------------------------------
#----------------------Alternating Group--------------------
#-----------------------------------------------------------
def transCount(A):
    cLen = 0
    k = 0
    n = len(A)
    iden = [i+1 for i in range(n)]
    if (A == iden):
        return 1
    pairs = [[0]*2 for i in range(n)]
    cycles = [[] for i in range(n)]
    for i in range(n):
        pairs[i][0] = A[i]
        pairs[i][1] = i+1
    for i in range(n-1):
        for j in range(n):
            if (pairs[i][1] == pairs[j][0]):
                temp = pairs[i+1]
                pairs[i+1] = pairs[j]
                pairs[j] = temp
    for i in range(n-1):
        if (pairs[i][1] == pairs[i+1][0]):
            cycles[k].append(pairs[i][0])
            cycles[k].append(pairs[i+1][0])
            cycles[k].append(pairs[i+1][1])
        else:
            k += 1
    cycles = [x for x in cycles if x != []]
    cycles.sort()
    cycles = list(cycles for cycles,_ in itertools.groupby(cycles))
    for i in range(len(cycles)):
        cycles[i].sort()
        cycles[i] = list(cycles[i] for cycles[i],_ in itertools.groupby(cycles[i]))
    for i in range(len(cycles)):
        cLen += len(cycles[i])-1
    if (cLen%2 == 0):
        return 1
    else:
        return 0            
def Alt(Sn):
    k = 0
    L = len(Sn)
    EvOd = [0]*L
    alt = [[] for i in range(L)]
    for i in range(L):
        if (transCount(Sn[i]) == 1):
            alt[k] = Sn[i]
            k += 1
    alt = [x for x in alt if x != []]
    return alt
#-----------------------------------------------------------
#-------------------------AltPicture------------------------
#-----------------------------------------------------------
def altPicture(alt, cPic, Sn, colors, sz, D):
    k = 0
    altLen = len(alt)
    altColors = [() for i in range(altLen)]
    N = len(Sn)
    im = Image.new('RGB', (sz,sz), color='white')
    px = im.load()
    for i in range(altLen):
        for j in range(N):
            if (alt[i] == Sn[j]):
                altColors[k] = tuple(colors[j])
                k += 1
                break
    for y in range(sz):
        for x in range(sz):
            if (cPic[x,y] == (0,0,0)):
                px[x,y] = cPic[x,y]
            elif (cPic[x,y] in altColors):
                px[x,y] = cPic[x,y]
            else:
                px[x,y] = (137,137,137)
    for y in range(D, sz):
        if (px[1,y] == (137,137,137)):
            for i in range(D,sz):
                if (px[i,y] in altColors):
                    px[i,y] = (137,137,137)
    for x in range(D,sz):
        if (px[x,1] == (137,137,137)):
            for i in range(D,sz):
                if (px[x,i] in altColors):
                    px[x,i] = (137,137,137)
    im.save('alt.png')
#-----------------------------------------------------------
#-------------------------OrbPicture------------------------
#-----------------------------------------------------------
def orbPicture(orb, cPic, Sn, colors, sz, D):
    k = 0
    orbLen = len(orb)
    orbColors = [() for i in range(orbLen)]
    N = len(Sn)
    im = Image.new('RGB', (sz,sz), color='white')
    px = im.load()
    for i in range(orbLen):
        for j in range(N):
            if (orb[i] == Sn[j]):
                orbColors[k] = tuple(colors[j])
                k += 1
                break
    for y in range(sz):
        for x in range(sz):
            if (cPic[x,y] == (0,0,0)):
                px[x,y] = cPic[x,y]
            elif (cPic[x,y] in orbColors):
                px[x,y] = cPic[x,y]
            else:
                px[x,y] = (137,137,137)
    im.save('orb.png')
#-----------------------------------------------------------
#-------------------------StabPicture-----------------------
#-----------------------------------------------------------
def stabPicture(stab, cPic, Sn, colors, sz, D):
    k = 0
    stabLen = len(stab)
    stabColors = [() for i in range(stabLen)]
    N = len(Sn)
    im = Image.new('RGB', (sz,sz), color='white')
    px = im.load()
    for i in range(stabLen):
        for j in range(N):
            if (stab[i] == Sn[j]):
                stabColors[k] = tuple(colors[j])
                k += 1
                break
    for y in range(sz):
        for x in range(sz):
            if (cPic[x,y] == (0,0,0)):
                px[x,y] = cPic[x,y]
            elif (cPic[x,y] in stabColors):
                px[x,y] = cPic[x,y]
            else:
                px[x,y] = (137,137,137)
    for y in range(D, sz):
        if (px[1,y] == (137,137,137)):
            for i in range(D,sz):
                if (px[i,y] in stabColors):
                    px[i,y] = (137,137,137)
    im.save('stab.png')
#-----------------------------------------------------------
#-------------------------SubPicture------------------------
#-----------------------------------------------------------
def subPicture(cPic, subs, subNum, Sn, colors, sz, D):
    k = 0
    L = len(subs)
    N = len(Sn)
    for i in range(L):
        print(i+1, len(subs[i]))
    im = Image.new('RGB', (sz, sz), color='white')
    px = im.load()
    subLen = len(subs[subNum-1])
    subG = subs[subNum-1]
    subColors = [() for i in range(subLen)]
    for i in range(subLen):
        for j in range(N):
            if (subG[i] == Sn[j]):
                subColors[k] = tuple(colors[j])
                k += 1
                break
    for y in range(sz):
        for x in range(sz):
            if (cPic[x,y] == (0,0,0)):
                px[x,y] = cPic[x,y]
            elif (cPic[x,y] in subColors):
                px[x,y] = cPic[x,y]
            else:
                px[x,y] = (137,137,137)
    for y in range(D, sz):
        if (px[1,y] == (137,137,137)):
            for i in range(D, sz):
                if (px[i,y] in subColors):
                    px[i,y] = (137,137,137)
    im.save('sub.png')
#-----------------------------------------------------------
#---------------------------Main----------------------------
#-----------------------------------------------------------
def main():
    n = int(input("Enter n: "))
    A = [i for i in range(1,n+1)]
    Sn = permutations(A)
    Sn = [list(x) for x in Sn]
    print(Sn[7])
    L = len(Sn)
    ##Coordinates = get_Coors(Sn, n)
    ##Angles = get_Angles(Coordinates, n)
    #Angles = failAngles(Sn, n)
    ##sd1 = SD(Angles1)
    ##sd2 = SD(Angles2)
    #Colors = get_Colors(Angles)
    #CayleyTable = get_Table(Sn, n)
    #subs = Subs(Sn)
    #subs = Cyclics(Sn)
    #stab = Stabilizer(CayleyTable)
    #orb = Orbit(CayleyTable)
    #alt = Alt(Sn)
    #cPic = MakePicture(Colors, CayleyTable, Sn)
    #subPic = subPicture(cPic[0], subs[0], subs[1], Sn, Colors, cPic[1], cPic[2])
    #stabPic = stabPicture(stab, cPic[0], Sn, Colors, cPic[1], cPic[2])
    #orbPic = orbPicture(orb, cPic[0], Sn, Colors, cPic[1], cPic[2])
    #altPic = altPicture(alt, cPic[0], Sn, Colors, cPic[1], cPic[2])
main()
