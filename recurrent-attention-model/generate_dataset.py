import numpy as np
from mnist_util import read
import operator as op
from PIL import Image
import math
from itertools import chain
import random
import itertools

colors = ['foreground light blue', 'foreground light green', 'foreground dark blue', 'foreground dark green']
bgcolors = ['background light blue', 'background light green', 'background dark blue', 'background dark green']
properties = ['color', 'bgcolor']

# returns a vector representation of an image
def extractImg(img, size=2):
    vec = []
    for element in img:
        element = realizeGrid(element)
        element = Image.fromarray(np.uint8(element*255)).convert('RGB')
        newvec = []
        imgwidth, imgheight = element.size
        width, height = imgwidth//size, imgheight//size
        for i in range(0, imgheight, height):
            for j in range(0, imgwidth, width):
                box = (j, i, j+height, i+width)
                im = element.crop(box)
                colors = Image.Image.getcolors(im)
                colors = sorted(colors, reverse=True)
                background, foreground = list(colors[0]), list(colors[1])
                background.pop(0)
                foreground.pop(0)
                newvec.append([round((x / 255), 1) for x in foreground[0]])
                newvec.append([round((x / 255), 1) for x in background[0]])
        vec.append(list(chain.from_iterable(newvec)))
    return vec

# returns a list of PIL images
def getImg(imgarray):
    img = []
    for element in imgarray:
        element = realizeGrid(element)
        element = Image.fromarray(np.uint8(element*255)).convert('RGB')
        img.append(element)
    return img

def generateRandomDataset(size):
    arr = []
    for k in range(size):
        gridImg = generateGridImg()
        arr.append(gridImg)
    return arr

def generateGridImg(size=2):
    while True:
        img = []
        for j in range(size**2):
            cell = {}
            cell['number'] = np.random.randint(10)
            cell['color'] = colors[np.random.randint(len(colors))]
            if colors.index(cell['color']) < len(colors)/2:
                cell['bgcolor'] = bgcolors[np.random.randint(len(bgcolors)/2, len(bgcolors))]
            else:
                cell['bgcolor'] = bgcolors[np.random.randint(len(bgcolors)/2)]
            img.append(cell)
        ## check if all targets are unique in the image, numbers excluded
        s = set()
        for e in img:
            s.add(frozenset((e['bgcolor'], e['color'])))
        if len(img) == len(s):
            break
    return img

# generates a complete data set of unique images (digits excluded)
def generateDataset():
    l = []
    cells = []
    for e in colors[0:len(colors)//2]:
       # print(e)
        for e1 in bgcolors[len(bgcolors)//2:len(bgcolors)]:
            cell = {}
            cell['number'] = np.random.randint(10)
            cell['color'] = e
            cell['bgcolor'] = e1
            cells.append(cell)
    for e in colors[len(colors)//2:len(colors)]:
        for e1 in bgcolors[0:len(bgcolors)//2]:
            cell = {}
            cell['number'] = np.random.randint(10)
            cell['color'] = e
            cell['bgcolor'] = e1
            cells.append(cell)
    l = [list(e) for e in itertools.permutations(cells, 4)]
    random.shuffle(l)
    return l

def initTargetMap(size=2):
    targetMap = []
    for i in range(int(size**2)):
        targetMap.append(True)
    return targetMap

def getChecklist(gridImg, targetMap):
    checklist = set()
    for i in range(len(gridImg)):
        if targetMap[i]:
            for k in properties:
                s = gridImg[i][k].split()
                checklist.add(s[0]+' '+s[1]+' '+s[2])
                checklist.add(s[0]+' '+s[2])
                checklist.add(s[0]+' '+s[1])
                checklist.add(s[1]+' '+s[2])
                checklist.add(s[2])
    return checklist

def noChecklist(checklist):
    for k in checklist:
        if len(checklist[k]) != 0:
            return False
    return True

def getTargets(gridImg, targetMap):
    targetIndices = []
    for i in range(len(gridImg)):
        for j in range(len(gridImg[i])):
            if targetMap[i][j]:
                targetIndices.append((i, j))
    return targetIndices

def countTargets(gridImg, targetMap, val):
    count = 0
    targetIndices = []
    for i in range(len(gridImg)):
        attributeExists = True
        if any(e not in gridImg[i]['color'] for e in val.split()):
            attributeExists = False
        if not attributeExists:
            if all(e in gridImg[i]['bgcolor'] for e in val.split()):
                attributeExists = True
        if targetMap[i] and attributeExists:
            count += 1
            targetIndices.append(i)
    return count, targetIndices

def selectSubTargetMap(gridImg, targetMap, val, reverse=False):
    for i in range(len(gridImg)):
        attributeExists = True
        if any(e not in gridImg[i]['color'] for e in val.split()):
            attributeExists = False
        if not attributeExists:
            if all(e in gridImg[i]['bgcolor'] for e in val.split()):
                attributeExists = True
        if targetMap[i] and not attributeExists and not reverse:
            targetMap[i] = False
        elif targetMap[i] and attributeExists and reverse:
            targetMap[i] = False

def reinitTargetMap(targetMap):
    for i in range(len(targetMap)):
        for j in range(len(targetMap[i])):
            targetMap[i][j] = True


# # QA instance generation

def initTarNum(gridImg):
    i = np.random.randint(len(gridImg))
    j = np.random.randint(len(gridImg))
    tarNumIndice = (i, j)
    return tarNumIndice

def getNumMap(targetMap):
    count = 0
    for i in range(len(targetMap)):
        if targetMap[i] == True:
            count = count + 1
    return count

def genQA_a(gridImg, targetMap, checklist, tarNum):
    while True:
        val = random.sample(checklist, 1)[0]
        count, targetIndices = countTargets(gridImg, targetMap, val)
        if getNumMap(targetMap) > count:
            break
    if not (tarNum in targetIndices):
        selectSubTargetMap(gridImg, targetMap, val, True)
    else:
        selectSubTargetMap(gridImg, targetMap, val)
    checklist = getChecklist(gridImg, targetMap)
    return val, checklist

# # QA sequence generation

def generateSeqQA(gridImg, tarNum):
    targetMap = initTargetMap(math.sqrt(len(gridImg)))
    checklist = getChecklist(gridImg, targetMap)
    aqmap ={
        'foreground light blue': [4, 2, 0, 6],
        'foreground light green': [4, 2, 1, 6],
        'foreground dark blue': [4, 3, 0, 6],
        'foreground dark green': [4, 3, 1, 6],
        'background light blue': [5, 2, 0, 6],
        'background light green': [5, 2, 1, 6],
        'background dark blue': [5, 3, 0, 6],
        'background dark green': [5, 3, 1, 6],
        'light blue': [2, 0, 6],
        'light green': [2, 1, 6],
        'dark blue': [3, 0, 6],
        'dark green': [3, 1, 6],
        'blue': [0, 6],
        'green': [1, 6],
        'foreground blue': [4, 0, 6],
        'foreground green': [4, 1, 6],
        'background blue': [5, 0, 6],
        'background green': [5, 1, 6],
        'foreground light': [4, 2, 6],
        'foreground dark': [4, 3, 6],
        'background light': [5, 2, 6],
        'background dark': [5, 3, 6]
    }
    q = []
    while True:
        q1, checklist = genQA_a(gridImg, targetMap, checklist, tarNum)
        q.append(q1)
        if getNumMap(targetMap) == 1:
            break
    q1 = []
    for i in range(len(q)):
        val = aqmap.get(q[i])
        for e in val:
            q1.append(e)
    q1.append(7)
    return q1


# # Image realization

import matplotlib.pyplot as plt
from scipy.ndimage.morphology import grey_dilation

def getNumPools():
    pools = {}
    for dataset in ['training', 'testing']:
        mnist = read(path='data/mnist')
        mnist = sorted(mnist, key = op.itemgetter(0))
        mnist = [(x[0], x[1].astype('float32')/255) for x in mnist]

        numPools= []
        for i in range(9):
            count = 0
            for j in range(len(mnist)):
                if mnist[j][0] != i:
                    break
                count+=1
            numPools.append(mnist[:count])
            mnist = mnist[count:]
        numPools.append(mnist)

        pools[dataset] = numPools
    return pools

numPools = getNumPools()
colorMap = {}
colorMap['foreground light blue'], colorMap['background light blue'] = [0, 0, 255], [0, 0, 255]
colorMap['foreground light green'], colorMap['background light green'] = [0, 255, 0], [0, 255, 0]
colorMap['foreground dark blue'], colorMap['background dark blue'] = [0, 0, 51], [0, 0, 51]
colorMap['foreground dark green'], colorMap['background dark green'] = [0, 51, 0], [0, 51, 0]

for k in colorMap:
    colorMap[k] = np.array(colorMap[k], dtype='float32').reshape((1, 1, 3)) / 255

def realizeSingleNumber(info, size = 28, dataset='training'):
    palette = np.ones((size, size, 3), dtype='float32') * colorMap[info['bgcolor']]

    num_sample_idx = np.random.randint(len(numPools[dataset][info['number']]))
    num_sample = numPools[dataset][info['number']][num_sample_idx][1]

    mask = num_sample.reshape((size, size, 1))
    palette = palette * (1-mask) + (mask * colorMap[info['color']]) * mask

    return palette

def realizeGrid(gridImg, size=28, dataset='training'):
    img_size = int(math.sqrt(len(gridImg)))
    img = np.zeros((size*img_size, size*img_size, 3))
    for i in range(img_size):
        for j in range(img_size):
            img[i*size:(i+1)*size, j*size:(j+1)*size, :] = realizeSingleNumber(gridImg[i*2+j], size=size, dataset=dataset)
    return img