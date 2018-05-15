import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
from scipy import spatial
import tensorflow as tf
import globalV
from loadData import loadData
from softmax import softmax
from classify import classify
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# For plotting
import plotly.plotly as py
import plotly.graph_objs as go
# py.sign_in('krittaphat.pug', 'oVTAbkhd2RQvodGOwrwp') # G-mail link
# py.sign_in('amps1', 'Z1KAk8xiPUyO2U58JV2K') # kpugdeet@syr.edu/12345678
py.sign_in('amps2', 'jGQHMBArdACog36YYCAI') # yli41@syr.edu/12345678
# py.sign_in('amps3', '5geLaNJlswmzDucmKikR') # liyilan0120@gmail.com/12345678


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset Path
    parser.add_argument('--BASEDIR', type=str, default='/media/dataHD3/kpugdeet/', help='Base folder for dataset and logs')
    parser.add_argument('--AWA2PATH', type=str, default='AWA2/Animals_with_Attributes2/', help='Path for AWA2 dataset')
    parser.add_argument('--CUBPATH', type=str, default='CUB/CUB_200_2011/', help='Path for CUB dataset')
    parser.add_argument('--SUNPATH', type=str, default='SUN/SUNAttributeDB/', help='Path for SUN dataset')
    parser.add_argument('--APYPATH', type=str, default='APY/attribute_data/', help='Path for APY dataset')
    parser.add_argument('--GOOGLE', type=str, default='PRE/GoogleNews-vectors-negative300.bin', help='Path for google Word2Vec model')
    parser.add_argument('--KEY', type=str, default='APY',help='Choose dataset (AWA2, CUB, SUN, APY)')
    parser.add_argument('--DIR', type=str, default='APY_0', help='Choose working directory')

    # Image size
    parser.add_argument('--width', type=int, default=227, help='Width')
    parser.add_argument('--height', type=int, default=227, help='Height')

    # Hyper Parameter
    parser.add_argument('--maxSteps', type=int, default=1, help='Number of steps to run trainer.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--numClass', type=int, default=32, help='Number of class')
    parser.add_argument('--batchSize', type=int, default=32, help='Batch size')
    parser.add_argument('--numAtt', type=int, default=300, help='Dimension of Attribute')

    # Initialize or Restore Model
    parser.add_argument('--TD', type=int, default=0, help='Train/Restore Darknet')
    parser.add_argument('--TA', type=int, default=0, help='Train/Restore Attribute')
    parser.add_argument('--TC', type=int, default=0, help='Train/Restore Classify')

    # Choose what to do
    parser.add_argument('--OPT', type=int, default=4, help='1.Darknet, 2.Attribute, 3.Classify, 4.Accuracy')
    parser.add_argument('--SELATT', type=int, default=1, help='1.Att, 2.Word2Vec, 3.Att+Word2Vec')
    globalV.FLAGS, _ = parser.parse_known_args()

    # Check Folder exist
    if not os.path.exists(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR):
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR)
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/softmax')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/softmax/logs')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/softmax/model')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/classify')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/classify/logs')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/classify/model')

    # Load data
    print('\nLoad Data for {0}'.format(globalV.FLAGS.KEY))
    (trainClass, trainAtt, trainVec, trainX, trainY, trainYAtt), (valClass, valAtt, valVec, valX, valY, valYAtt), (testClass, testAtt, testVec, testX, testY, testYAtt) = loadData.getData()
    if globalV.FLAGS.KEY == 'SUN' or globalV.FLAGS.KEY == 'APY':
        print('       {0:>10} {1:>12} {2:>10} {3:>20} {4:>10} {5:>12}'.format('numClass', 'classAtt', 'classVec','inputX', 'outputY', 'outputAtt'))
        print('Train: {0:>10} {1:>12} {2:>10} {3:>20} {4:>10} {5:>12}'.format(len(trainClass), str(trainAtt.shape), str(trainVec.shape), str(trainX.shape), str(trainY.shape), str(trainYAtt.shape)))
        print('Valid: {0:>10} {1:>12} {2:>10} {3:>20} {4:>10} {5:>12}'.format(len(valClass), str(valAtt.shape), str(valVec.shape), str(valX.shape), str(valY.shape), str(valYAtt.shape)))
        print('Test:  {0:>10} {1:>12} {2:>10} {3:>20} {4:>10} {5:>12}'.format(len(testClass), str(testAtt.shape), str(testVec.shape), str(testX.shape), str(testY.shape), str(testYAtt.shape)))
    else:
        print('       {0:>10} {1:>12} {2:>10} {3:>20} {4:>10}'.format('numClass', 'classAtt', 'classVec','inputX', 'outputY'))
        print('Train: {0:>10} {1:>12} {2:>10} {3:>20} {4:>10}'.format(len(trainClass), str(trainAtt.shape), str(trainVec.shape), str(trainX.shape), str(trainY.shape)))
        print('Valid: {0:>10} {1:>12} {2:>10} {3:>20} {4:>10}'.format(len(valClass), str(valAtt.shape), str(valVec.shape), str(valX.shape), str(valY.shape)))
        print('Test:  {0:>10} {1:>12} {2:>10} {3:>20} {4:>10}'.format(len(testClass), str(testAtt.shape), str(testVec.shape), str(testX.shape), str(testY.shape)))

    # Show class name that index with total classes
    def printClassName(pos):
        if pos < len(trainClass):
            return trainClass[pos]
        elif pos < len(trainClass)+len(valClass):
            return valClass[pos-len(trainClass)]
        else:
            return testClass[pos-len(trainClass)-len(valClass)]

    # Attribute Modification
    concatAtt = np.concatenate((trainAtt, valAtt, testAtt), axis=0)
    word2Vec_concatAtt = np.concatenate((trainVec, valVec, testVec), axis=0)
    combineAtt = np.concatenate((concatAtt, word2Vec_concatAtt), axis=1)

    if globalV.FLAGS.SELATT == 1:
        concatAtt_D = concatAtt
    elif globalV.FLAGS.SELATT == 2:
        concatAtt_D = word2Vec_concatAtt
    else:
        concatAtt_D = combineAtt

    # Check where there is some class that has same attributes
    print('\nCheck matching classes attributes')
    for i in range(concatAtt_D.shape[0]):
        for j in range(i + 1, concatAtt_D.shape[0]):
            if np.array_equal(concatAtt_D[i], concatAtt_D[j]):
                print('{0} {1}: {2} {3}'.format(i, printClassName(i), j, printClassName(j)))
    print('')

    print('\nMerge Data')
    allX = np.concatenate((trainX, valX, testX), axis=0)
    allY = np.concatenate((trainY, valY+len(trainClass), testY+len(trainClass)+len(valClass)), axis=0)
    s = np.arange(allX.shape[0])
    np.random.seed(100)
    np.random.shuffle(s)
    allX = allX[s]
    allY = allY[s]

    # Split Data
    trX = None; vX = None; teX = None
    trY = None; vY = None; teY = None
    for z in range(0, globalV.FLAGS.numClass):
        eachInputX = []
        eachInputY = []
        for k in range(0, allX.shape[0]):
            if allY[k] == z:
                eachInputX.append(allX[k])
                eachInputY.append(allY[k])
        eachInputX = np.array(eachInputX)
        eachInputY = np.array(eachInputY)
        div1 = int(eachInputX.shape[0] * 0.7)
        div2 = int(eachInputX.shape[0] * 0.8)
        if trX is None:
            trX = eachInputX[:div1]
            vX = eachInputX[div1:div2]
            teX = eachInputX[div2:]
            trY = eachInputY[:div1]
            vY = eachInputY[div1:div2]
            teY = eachInputY[div2:]
        else:
            trX = np.concatenate((trX,  eachInputX[:div1]), axis=0)
            vX = np.concatenate((vX, eachInputX[div1:div2]), axis=0)
            teX = np.concatenate((teX, eachInputX[div2:]), axis=0)
            trY = np.concatenate((trY, eachInputY[:div1]), axis=0)
            vY = np.concatenate((vY, eachInputY[div1:div2]), axis=0)
            teY = np.concatenate((teY, eachInputY[div2:]), axis=0)


    print('Shuffle Data shape')
    print(trX.shape, trY.shape)
    print(vX.shape, vY.shape)
    print(teX.shape, teY.shape)

    baX = None; baY = None
    # Balance training class
    sampleEach = 500
    for z in range(globalV.FLAGS.numClass):
        eachInputX = []
        eachInputY = []
        for k in range(0, trX.shape[0]):
            if trY[k] == z:
                eachInputX.append(trX[k])
                eachInputY.append(trY[k])
        eachInputX = np.array(eachInputX)
        eachInputY = np.array(eachInputY)

        if eachInputX.shape[0] > sampleEach:
            if baX is None:
                baX = eachInputX[:sampleEach]
                baY = eachInputY[:sampleEach]
            else:
                baX = np.concatenate((baX, eachInputX[:sampleEach]), axis=0)
                baY = np.concatenate((baY, eachInputY[:sampleEach]), axis=0)
        else:
            duX = np.copy(eachInputX)
            duY = np.copy(eachInputY)
            while duX.shape[0] < sampleEach:
                duX = np.concatenate((duX, eachInputX), axis=0)
                duY = np.concatenate((duY, eachInputY), axis=0)
            if baX is None:
                baX = duX[:sampleEach]
                baY = duY[:sampleEach]
            else:
                baX = np.concatenate((baX, duX[:sampleEach]), axis=0)
                baY = np.concatenate((baY, duY[:sampleEach]), axis=0)

    # Get all Classes name and attribute name
    allClassName = np.concatenate((np.concatenate((trainClass, valClass), axis=0), testClass), axis=0)
    with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.APYPATH + 'attribute_names.txt', 'r') as f:
        allClassAttName = [line.strip() for line in f]

    if globalV.FLAGS.OPT == 2:
        print('\nTrain Alexnet')
        softmaxModel = softmax()
        softmaxModel.train(baX, baY, vX, vY, teX, teY)

    elif globalV.FLAGS.OPT == 4:
        print('\nClassify')
        softmaxModel = softmax()
        predY = softmaxModel.getPred(trX)
        print('Train Accuracy = {0:.4f}%'.format(np.mean(np.equal(predY, trY))*100))
        predY = softmaxModel.getPred(vX)
        print('Val Accuracy = {0:.4f}%'.format(np.mean(np.equal(predY, vY)) * 100))
        predY = softmaxModel.getPred(teX)
        print('Test Accuracy = {0:.4f}%'.format(np.mean(np.equal(predY, teY)) * 100))

        # Accuracy for each class and Confusion matrix
        print('')
        confusion = []
        for z in range(globalV.FLAGS.numClass):
            eachInputX = []
            eachInputY = []
            for k in range(0, teX.shape[0]):
                if teY[k] == z:
                    eachInputX.append(teX[k])
                    eachInputY.append(teY[k])
            eachInputX = np.array(eachInputX)
            eachInputY = np.array(eachInputY)
            predY = softmaxModel.getPred(eachInputX)
            print('Class: {0:<15} Size: {1:<10} Accuracy = {2:.4f}%'.format(printClassName(z), eachInputX.shape[0], np.mean(np.equal(predY, eachInputY)) * 100))

            # Confusion matrix
            tmpScore = softmaxModel.getScore(eachInputX)
            tmpSort = np.argsort(-tmpScore, axis=1)
            tmpPred = tmpSort[:, :1]
            tmpPred = np.reshape(tmpPred, -1)
            tmpCountEach = np.bincount(tmpPred, minlength=32)
            tmpCountEach = np.array(tmpCountEach)/(eachInputX.shape[0]*1)
            confusion.append(tmpCountEach)

        confusion = np.array(confusion)
        # Confusion Matrix
        tmpClassName = [printClassName(x) for x in range(globalV.FLAGS.numClass)]
        trace = go.Heatmap(z=confusion,
                           x=tmpClassName,
                           y=tmpClassName,
                           zmax=1.0,
                           zmin=0.0
                           )
        data = [trace]
        layout = go.Layout(title=globalV.FLAGS.KEY, width=1920, height=1080,
                           yaxis=dict(
                               ticktext=tmpClassName,
                               tickvals=np.arange(len(tmpClassName))))
        fig = go.Figure(data=data, layout=layout)
        py.image.save_as(fig, filename=globalV.FLAGS.DIR + '_Confusion_' + str(1) + '.png')











