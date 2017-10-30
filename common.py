import re
def loadTrainingData(filepath="training.txt"):
    # Load each line of the training data into an array.
    verses = []
    with open(filepath, "r", encoding="latin-1") as file:
        for line in file:
            verses.append(line)
    # Pair up each couple of lines so that each line has both the first and second half of each verse.
    versePairs = [i+j for i,j in zip(verses[::2], verses[1::2])]
    # Remove verses with \x97 (hyphen telling singers to skip a note) as they don't occur often enough for the model to learn
    # how they work.
    versePairs = [verse[:-1] for verse in versePairs if "\x97" not in verse]
    return versePairs
# This function will put all important features (|, · and *) to be single characters with no whitespace.
def regexClean(verse):
    # Replace "- | " with "|"
    verse = re.sub(r"\- \| ", r"|", verse)
    # Replace "- · " with "·"
    verse = re.sub(r"\- · ","·", verse)
    # Replace " | " with " |"
    verse = re.sub(r" \| ", r" |", verse)
    # Replace " · " with " ·"
    verse = re.sub(r" · ", " ·", verse)
    # Replace " *\n" with "* "
    verse = re.sub(r" \*\n", " *", verse)
    return verse
# This function just reverses regexClean.
def regexUnclean(verse):
    # Replace " |" with " | "
    verse = re.sub(r"(?<= )\|", r"| ", verse)
    # Replace " ·" with " · "
    verse = re.sub(r"(?<= )·", "· ", verse)
    # Replace letter|letter with letter- | letter
    verse = re.sub(r"(?<=[a-zA-Z])\|(?=[a-zA-Z])", "- | ", verse)
    # Replace "* " with " *\n"
    verse = re.sub(r" \*", r" *\n", verse)
    # Replace letter·letter with letter- · letter
    verse = re.sub(r"(?<=[a-zA-Z])·(?=[a-zA-Z])", "- · ", verse)
    return verse
# This function creates two strings, one is the original string with the selected chars removed,
# the second is a string with a character if the original string has that character inserted after that point
# and 0 otherwise.
def splitString(string, chars=["|","·","*"]):
    stringOut = []
    toMergeOut = []
    for char in string:
        if char in chars:
            toMergeOut[-1] = char
        else:
            stringOut.append(char)
            toMergeOut.append("0")
    return "".join(stringOut), "".join(toMergeOut)
# This function reverses splitString by combining the two strings it produces into one string again.
def mergeStrings(string, toMerge):
    out = []
    toMerge = toMerge[-len(string):]
    for (a, b) in zip(string, toMerge):
        out.append(a)
        if b != '0':
            out.append(b)
    return "".join(out)
def cleanAndSplitVerses(versePairs):
    # Tokenise all important characters then split them into X and Y (i.e. input and desired output.)
    cleanedVersePairs = [regexClean(verse) for verse in versePairs]
    splitVerses = [splitString(verse) for verse in cleanedVersePairs]
    X = [tup[0] for tup in splitVerses]
    Y = [tup[1] for tup in splitVerses]
    return X, Y
# Perform a custom one hot encoding on the input data. There is a special bit to indicate if a character is capitalised.
alphabet=" abcdefghijklmnopqrstuvwxyz"
numbers = set(list("0123456789"))
spaces = set(list(" -–"))
pairedPunctuation = set(list("\"\'“”‘’()"))
otherPunctuation = set(list("!?,.:;"))
charToInt = dict((c, i) for i, c in enumerate(alphabet))
for n in numbers:
    charToInt[n] = 27
for s in spaces:
    charToInt[s] = 0
for p in pairedPunctuation:
    charToInt[p] = 28
for o in otherPunctuation:
    charToInt[o] = 29
def enc(char):
    letter = [0 for _ in range(31)]
    letter[charToInt[char.lower()]] = 1
    if char.isupper():
        letter[30] = 1
    return letter
def encString(string):
    res = []
    for char in list(string):
        res.append(enc(char))
    return res
# Perform a one hot encoding on the target data.
yToEnc = {"0": [1, 0, 0, 0], "|": [0, 1, 0, 0], "·": [0, 0, 1, 0], "*": [0, 0, 0, 1]}
def encY(toMerge):
    res = []
    for char in list(toMerge):
        res.append(yToEnc[char])
    return res
def decY(array):
    res = []
    for l in array:
        if l[0] == 1:
            res.append("|")
        elif l[1] == 1:
            res.append("·")
        elif l[2] == 1:
            res.append("*")
        else:
            res.append("0")
    return "".join(res)
def encXandY(X, Y):
    Xenc = [encString(v) for v in X]
    Yenc = [encY(tm) for tm in Y]
    return Xenc, Yenc
from keras.preprocessing.sequence import pad_sequences
def padXandY(Xenc, Yenc, maxlen):
    Xnp = pad_sequences(Xenc, maxlen=maxlen)
    Ynp = pad_sequences(Yenc, maxlen=maxlen, value=[1,0,0,0])
    return Xnp, Ynp
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
from keras import backend as K
import numpy as np
# This next function doesn't work as Numpy operations can't be compiled for use on GPU.
def custom_categorical_crossentropy(y_true, y_pred):
    y_pred_np = K.eval(y_pred)
    y_semi_true_np = np.zeros_like(y_pred_np)
    y_semi_true_np[:,0] = 1
    # Find the most likely mid verse split point before -12th position
    split = np.argmax(y_pred_np[:-12,3], axis=0)
    firstHalfBars = np.argpartition(y_pred_np[:split,1], -2)[-2:]
    secondHalfBars = (np.argpartition(y_pred_np[split+1:,1], -3) + split + 1)[-3:]
    def add_dot(bar_index, next_bar_index, dots):
        dot_prob = np.max(y_pred_np[bar_index+1:next_bar_index,2], axis=0)
        if dot_prob >= 0.25:
            dots.append(np.argmax(y_pred_np[bar_index+1:next_bar_index,2], axis=0))
        return dots
    dots = []
    dots = add_dot(firstHalfBars[1], firstHalfBars[2], dots)
    dots = add_dot(secondHalfBars[1], secondHalfBars[2], dots)
    dots = add_dot(secondHalfBars[2], secondHalfBars[3], dots)
    y_semi_true_np[split,:] = np.array([0,0,0,1])
    for index in firstHalfBars + secondHalfBars:
        y_semi_true_np[index,:] = np.array([0,1,0,0])
    for index in dots:
        y_semi_true_np[index,:] = np.array([0,0,1,0])
    return K.categorical_crossentropy(y_semi_true_np, y_pred)
# Convert the model.predict_classes result back into a toMerge string.
def decClasses(array):
    res = []
    for l in array:
        if l == 1:
            res.append("|")
        elif l == 2:
            res.append("·")
        elif l == 3:
            res.append("*")
        else:
            res.append("0")
    return "".join(res)
# Convert probability arrays into class values then decode with decClasses.
def getClasses(row):
    predictions = np.argmax(row, axis=1)
    return decClasses(predictions)
# Perform a more intelligent prediction from the class probabilities enforcing the ||*||| format of a proper chant.
def getToComb(row):
    # Make a result array of the correct shape.
    predictions = np.zeros(row.shape[0])
    # Find the most likely mid-verse split point.
    split = np.argmax(row[:,3])
    predictions[split] = 3
    # Find the two most likely indices of the first half for bar lines.
    firstHalfBars = np.argpartition(row[:split,1], -2)[-2:]
    # Find the three most likely indices of the second half for bar lines.
    secondHalfBars = (np.argpartition(row[split+1:,1], -3) + split + 1)[-3:]
    # Find where a dot is most likely (i.e. the same dot system as before).
    dots = np.argwhere(np.argmax(row, axis=1) == 2)
    predictions[firstHalfBars] = 1
    predictions[secondHalfBars] = 1
    predictions[dots] = 2
    return decClasses(predictions)
