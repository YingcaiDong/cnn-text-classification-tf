import numpy as np
import re
import itertools
from collections import Counter


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
    空白符\s，两个至多个
    '''
    string = re.sub(r"\s{2,}", " ", string)
    # strip default token: space, hence split words with space, convert to lowercase.
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    '''
        For each line of comment, strip lead and trail white space
    '''
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    '''
        Join a sequence of arrays along an existing axis.
        The arrays must have the same shape, except in the dimension 0.
    '''
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


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
