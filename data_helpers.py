import numpy as np
import re
import pandas as pd


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    '''
    replace some characters with " " except A-Za-z0-9(),!?'`
    '''
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    '''
    add space to all these 'abbreviation
    '''
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    '''
    surround some punctuation with space.
    '''
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    '''
    empty space \s，maximum 2 empty space
    '''
    string = re.sub(r"\s{2,}", " ", string)
    # strip default token: space, hence strip front and end space, convert words to lowercase.
    return string.strip().lower()


def load_data_and_labels(data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    data = pd.read_csv(data_file)
    reviews = data["Text"]
    scores = data["Score"]
    '''
        For each line of comment, strip lead and trail white space
    '''
    reviews = [s.strip() for s in reviews]
    # Split by words
    x_text = [clean_str(sent) for sent in reviews]
    # Generate labels
    y_labels = []
    for index, score in enumerate(scores):
        y_labels.append({
            1: [1, 0, 0, 0, 0],
            2: [0, 1, 0, 0, 0],
            3: [0, 0, 1, 0, 0],
            4: [0, 0, 0, 1, 0],
            5: [0, 0, 0, 0, 1]
        }[score])
        if index%1000 == 0:
            print("Load " + str(index) + " data into the memory\n")
    '''
        Join a sequence of arrays along an existing axis.
        The arrays must have the same shape, except in the dimension 0.
    '''
    return [x_text, y_labels]


def batch_iter(data, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    # Shuffle the data at each epoch
    if shuffle:
        '''
        np.arrange()
        Return evenly spaced values within a given interval. Interval end by data_size, step by default = 1
        Examples
        --------
        >>> np.arange(3)
        array([0, 1, 2])
        ====================================
        np.random.permutation()
        Examples
        --------
        >>> np.random.permutation(10)
        >>> np.random.permutation([1, 4, 9, 12, 15])
        array([15,  1,  9,  4, 12])
        '''
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]
