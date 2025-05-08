from tensorflow.keras import layers
import numpy as np

def transform_data(normalizer, lookup_layers, cat_col, num_col, data):
    numeric_data = normalizer(data[num_col].to_numpy())
    categorical_data = [lookup_layers[col](data[col]) for col in cat_col]
    return layers.concatenate([numeric_data] + categorical_data)

def preprocess_data(cat_col, num_col, dataset, train=True):
    train_dataset = dataset.sample(frac=0.8, random_state=42)
    train_labels = train_dataset.pop('expenses')

    normalizer = layers.Normalization()
    normalizer.adapt(train_dataset[num_col].to_numpy())

    lookup_layers = {}
    for col in cat_col:
        layer = layers.StringLookup(output_mode='one_hot')
        layer.adapt(train_dataset[col])
        lookup_layers[col] = layer
    if train:
        data = transform_data(normalizer, lookup_layers, cat_col, num_col, train_dataset)
        labels = np.log(train_labels)
    else:
        test_dataset = dataset.drop(train_dataset.index)
        test_labels = test_dataset.pop('expenses')
        data = transform_data(normalizer, lookup_layers, cat_col, num_col, test_dataset)
        labels = np.log(test_labels)

    return data, labels




    

