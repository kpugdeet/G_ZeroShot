import tensorflow as tf
import numpy as np
import globalV
from alexnet import *


class softmax (object):
    def __init__(self):
        # All placeholder
        self.x = tf.placeholder(tf.float32, name='inputImage', shape=[None, globalV.FLAGS.height, globalV.FLAGS.width, 3])
        self.y = tf.placeholder(tf.int64, name='outputClassIndexSeen', shape=[None])
        self.isTraining = tf.placeholder(tf.bool)

        with tf.variable_scope("softmax"):
            wF_H = tf.get_variable(name='wF_H', shape=[1000, 10], dtype=tf.float32)
            bF_H = tf.get_variable(name='bF_H', shape=[10], dtype=tf.float32)

        # Input -> feature extract from CNN
        coreNet = alexnet(self.x)
        self.yOut = tf.add(tf.matmul(coreNet, wF_H), bF_H)

        # Loss
        self.totalLoss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(self.y, 10), logits=self.yOut))

        # Define correct Prediction and accuracy
        correctPrediction = tf.equal(tf.argmax(self.yOut, 1), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
        self.predY = tf.argmax(self.yOut, 1)

        # Define Optimizer
        updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(updateOps):
            self.trainStep = tf.train.AdamOptimizer(globalV.FLAGS.lr).minimize(self.totalLoss)

        # Log all output
        self.averageL = tf.placeholder(tf.float32)
        tf.summary.scalar("averageLoss", self.averageL)
        self.averageAcc = tf.placeholder(tf.float32)
        tf.summary.scalar("averageAcc", self.averageAcc)

        # Merge all log
        self.merged = tf.summary.merge_all()

        # Initialize session
        tfConfig = tf.ConfigProto(allow_soft_placement=True)
        tfConfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfConfig)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=None)

        # Log directory
        if globalV.FLAGS.TA == 1:
            if tf.gfile.Exists(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/softmax/logs'):
                tf.gfile.DeleteRecursively(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/softmax/logs')
            tf.gfile.MakeDirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/softmax/logs')
        self.trainWriter = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/softmax/logs/train', self.sess.graph)
        self.valWriter = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/softmax/logs/validate')
        self.testWriter = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/softmax/logs/test')

        # Start point and log
        self.Start = 1
        self.Check = 10
        if globalV.FLAGS.TA == 0:
            self.restoreModel()

    def restoreModel(self):
        self.saver.restore(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/softmax/model/model.ckpt')
        npzFile = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/softmax/model/checkpoint.npz')
        self.Start = npzFile['Start']
        self.Check = npzFile['Check']

    def train(self, trX, trY, vX, vY, teX, teY):
        for i in range(self.Start, globalV.FLAGS.maxSteps + 1):
            print('Loop {0}/{1}'.format(i, globalV.FLAGS.maxSteps))

            # Shuffle Data
            s = np.arange(trX.shape[0])
            np.random.shuffle(s)
            trX = trX[s]
            trY = trY[s]

            # Train
            losses = []
            accs = []
            for j in range(0, trX.shape[0], globalV.FLAGS.batchSize):
                xBatch = trX[j:j + globalV.FLAGS.batchSize]
                yBatch = trY[j:j + globalV.FLAGS.batchSize]
                trainLoss, trainAcc, _ = self.sess.run([self.totalLoss, self.accuracy, self.trainStep], feed_dict={self.x: xBatch, self.y: yBatch, self.isTraining: 1})
                losses.append(trainLoss)
                accs.append(trainAcc)
            feed = {self.averageL: sum(losses) / len(losses), self.averageAcc: sum(accs) / len(accs)}
            summary = self.sess.run(self.merged, feed_dict=feed)
            self.trainWriter.add_summary(summary, i)

            # Validation
            losses = []
            accs = []
            for j in range(0, vX.shape[0], globalV.FLAGS.batchSize):
                xBatch = vX[j:j + globalV.FLAGS.batchSize]
                yBatch = vY[j:j + globalV.FLAGS.batchSize]
                valLoss, valAcc = self.sess.run([self.totalLoss, self.accuracy], feed_dict={self.x: xBatch, self.y: yBatch, self.isTraining: 0})
                losses.append(valLoss)
                accs.append(valAcc)
            feed = {self.averageL: sum(losses) / len(losses), self.averageAcc: sum(accs) / len(accs)}
            summary = self.sess.run(self.merged, feed_dict=feed)
            self.valWriter.add_summary(summary, i)

            # Test
            losses = []
            accs = []
            for j in range(0, teX.shape[0], globalV.FLAGS.batchSize):
                xBatch = teX[j:j + globalV.FLAGS.batchSize]
                yBatch = teY[j:j + globalV.FLAGS.batchSize]
                teLoss, teAcc = self.sess.run([self.totalLoss, self.accuracy], feed_dict={self.x: xBatch, self.y: yBatch, self.isTraining: 0})
                losses.append(teLoss)
                accs.append(teAcc)
            feed = {self.averageL: sum(losses) / len(losses), self.averageAcc: sum(accs) / len(accs)}
            summary = self.sess.run(self.merged, feed_dict=feed)
            self.testWriter.add_summary(summary, i)

            if i % self.Check == 0:
                savePath = self.saver.save(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/softmax/model/model.ckpt',global_step=i)
                print('Model saved in file: {0}'.format(savePath))
                if i / self.Check == 10:
                    self.Check *= 10

            # Save State
            np.savez(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/softmax/model/checkpoint.npz', Start=i+1, Check=self.Check)
            self.saver.save(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/softmax/model/model.ckpt')

    def getPred(self, inputX):
        outputAtt = None
        for j in range(0, inputX.shape[0], globalV.FLAGS.batchSize):
            xBatch = inputX[j:j + globalV.FLAGS.batchSize]
            if outputAtt is None:
                outputAtt = self.sess.run(self.predY, feed_dict={self.x: xBatch, self.isTraining: 0})
            else:
                outputAtt = np.concatenate((outputAtt, self.sess.run(self.predY, feed_dict={self.x: xBatch, self.isTraining: 0})), axis=0)
        return outputAtt

    def getScore(self, inputX):
        outputAtt = None
        for j in range(0, inputX.shape[0], globalV.FLAGS.batchSize):
            xBatch = inputX[j:j + globalV.FLAGS.batchSize]
            if outputAtt is None:
                outputAtt = self.sess.run(self.yOut, feed_dict={self.x: xBatch, self.isTraining: 0})
            else:
                outputAtt = np.concatenate((outputAtt, self.sess.run(self.yOut, feed_dict={self.x: xBatch, self.isTraining: 0})), axis=0)
        return outputAtt
