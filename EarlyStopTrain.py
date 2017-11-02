import tensorflow as tf
from tensorflow.contrib.keras import callbacks as call

class Stop_training(call.EarlyStopping):
    __init__(
        monitor = 'val_loss',
        min_delta=0,
        patience=0,
        verbose=0,
        mode='auto'
    )
    call.EarlyStopping.__init__()