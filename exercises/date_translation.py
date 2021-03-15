from datetime import date
from numpy.random import randint

import numpy as np
import tensorflow as tf


# generate random dates

def random_dates(n, min_date, max_date):
    if isinstance(min_date, date):
        min_date = min_date.toordinal()
    if isinstance(max_date, date):
        max_date = max_date.toordinal()
    new_dates = randint(min_date, max_date, n)
    new_dates = [date.fromordinal(new_date) for new_date in new_dates]
    
    return new_dates

def generate_training_dates(n, input_format='%B %d, %Y', output_format='%Y-%m-%d', min_date=date(1000,1,1), max_date=date(9999,12,31)):
    dates = random_dates(n, min_date, max_date)
    x = [d.strftime(input_format) for d in dates]
    y = [d.strftime(output_format) for d in dates]
    return x, y

# generate data to train character seq2seq model

def str_to_id(date_str, chars):
    return [chars.index(c) for c in date_str]

def preprocess_dates(dates):
    CHARS = list(set(''.join(dates)))
    dates_id = [str_to_id(d, CHARS) for d in dates]
    X = tf.ragged.constant(dates_id, ragged_rank=1)
    return (X + 1).to_tensor()

def generate_training_data(n, **kwargs):
    x,y = generate_training_dates(n, **kwargs)
    return preprocess_dates(x), preprocess_dates(y)