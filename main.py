#Shayling Zhao

import math
import os
import nltk


def trainHam(hamText):
    # Creating a list of words/characters
    h_tokens = nltk.word_tokenize(hamText)
    # Getting length of the list
    vocHam = len(set(h_tokens))
    trainHamDict = {}
    # Keeping a running count of number of times words show up in the text
    for h in h_tokens:
        if h in trainHamDict:
            trainHamDict[h] += 1
        else:
            trainHamDict[h] = 1
    return trainHamDict


def trainSpam(spamText):
    # Creating a list of words/characters
    s_tokens = nltk.word_tokenize(spamText)
    # Getting length of the list
    vocSpam = len(set(s_tokens))
    trainSpamDict = {}
    # Keeping a running count of number of times words show up in the text
    for s in s_tokens:
        if s in trainSpamDict:
            trainSpamDict[s] += 1
        else:
            trainSpamDict[s] = 1
    return trainSpamDict


def testH(testHam, hamDict, spamDict):
    # Creating a list of words/characters
    h_tokens = nltk.word_tokenize(testHam)
    testHamDict = {}
    # Keeping a running count of number of times words show up in the text
    for h in h_tokens:
        if h in testHamDict:
            testHamDict[h] += 1
        else:
            testHamDict[h] = 1
    h_probHam = 0
    h_probSpam = 0
    # Checking for amount correct in Ham files
    for key in testHamDict:
        if key in hamDict:
            # Doing Laplace smoothing so adding 1 for every key found
            prob = (hamDict[key] + 1) / (sum(hamDict.values()) + len(hamDict.keys()))
            h_probHam += math.log(prob, 2)
        else:
            # technically 0 + 1 in numerator because 0 in hamDict
            prob = 1 / (sum(hamDict.values()) + len(hamDict.keys()))
            h_probHam += math.log(prob, 2)

        if key in spamDict:
            # Doing Laplace smoothing
            prob = (spamDict[key] + 1) / (sum(spamDict.values()) + len(spamDict.keys()))
            h_probSpam += math.log(prob, 2)
        else:
            prob = 1 / (sum(spamDict.values()) + len(spamDict.keys()))
            h_probSpam += math.log(prob, 2)
    return h_probHam, h_probSpam


def testS(testSpam, hamDict, spamDict):
    # Creating a list of words/characters
    h_tokens = nltk.word_tokenize(testSpam)
    testSpamDict = {}
    # Keeping a running count of number of times words show up in the text
    for h in h_tokens:
        if h in testSpamDict:
            testSpamDict[h] += 1
        else:
            testSpamDict[h] = 1
    # print(testSpamDict)
    # Checking for amount correct in Spam files
    s_probHam = 0
    s_probSpam = 0
    for key in testSpamDict:
        if key in hamDict:
            # Doing Laplace smoothing
            prob = (hamDict[key] + 1) / (sum(hamDict.values()) + len(hamDict.keys()))
            s_probHam += math.log(prob, 2)
        else:
            prob = 1 / (sum(hamDict.values()) + len(hamDict.keys()))
            s_probHam += math.log(prob, 2)

        if key in spamDict:
            # Doing Laplace smoothing
            prob = (spamDict[key] + 1) / (sum(spamDict.values()) + len(spamDict.keys()))
            s_probSpam += math.log(prob, 2)
        else:
            prob = 1 / (sum(spamDict.values()) + len(spamDict.keys()))
            s_probSpam += math.log(prob, 2)
    return s_probHam, s_probSpam


if __name__ == '__main__':

    #Loop through Ham folder in training and read in files
    hamText = "" #Initiating variable to avoide scope problem
    path_trainHam = './train/ham'
    for file in os.listdir(path_trainHam):
        if file.endswith('.txt'):
            with open('./train/ham/' + file, 'r', errors='ignore') as opened_file:
                # This is to read out all the text as one text to make one tokens list rather than several in trainHam
                hamText = hamText + " " + opened_file.read()

    #Loop through Spam folder in training and read in files
    spamText = "" #Initiating variable to avoid scope problem
    path_trainSpam = './train/spam'
    for file in os.listdir(path_trainSpam): #returns list of all the files in this path
        if file.endswith('.txt'):
            with open('./train/spam/' + file, 'r', errors='ignore') as opened_file:
                # This is to read out all the text as one text to make one tokens list rather than several in trainSpam
                spamText = spamText + " " + opened_file.read()

    # print(trainHam(hamText))
    # print(trainSpam(spamText))
    # print(" ")
    hamDict = trainHam(hamText)
    spamDict = trainSpam(spamText)
    correctSpam = 0
    correctHam = 0
    totalSpam = 0
    totalHam = 0
    # Loop through Spam folder in testing and read in files
    testHam = ""  # Initiating variable to avoid scope problem
    path_testHam = './test/ham'
    for file in os.listdir(path_testHam):  # returns list of all the files in this path
        if file.endswith('.txt'):
            with open('./test/ham/' + file, 'r', errors='ignore') as opened_file:
                # This is to read out all the text as one text to make one tokens list rather than several in trainSpam
                testHam = opened_file.read()
                probHam, probSpam = testH(testHam, hamDict, spamDict)
                if probHam > probSpam:
                    correctHam += 1
                totalHam += 1

    # Loop through Spam folder in testing and read in files
    testSpam = ""  # Initiating variable to avoid scope problem
    path_testSpam = './test/spam'
    for file in os.listdir(path_testSpam):  # returns list of all the files in this path
        if file.endswith('.txt'):
            with open('./test/spam/' + file, 'r', errors='ignore') as opened_file:
                # This is to read out all the text as one text to make one tokens list rather than several in trainSpam
                testSpam = opened_file.read()
                probHam, probSpam = testS(testSpam, hamDict, spamDict)
                if probSpam > probHam:
                    correctSpam += 1
                totalSpam += 1


    hamPercent = "{:.2f}".format(100*(correctHam/totalHam))
    spamPercent = "{:.2f}".format(100*(correctSpam/totalSpam))


    print("The percent correct in Ham testing data is about", hamPercent + "%")
    print("The percent correct in Spam testing data is about", spamPercent + "%")
