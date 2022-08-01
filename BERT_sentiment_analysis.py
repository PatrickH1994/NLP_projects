"""
In this file I create a sentiment classifier for hotel reviews by fine-tuning ALBERT. I take the following steps
1) prepare training data
- Data is taken from kaggle (https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews)

2) download albert classifier and preprocessor from tensorflow
- I use albert as the basline model because it's a lighter version of BERT and thus requires less computing power for training. (see: https://tfhub.dev/google/albert_base/3)

3) fine-tune the model
- When fine-tuning the model I freeze the weights for the baseline model to reduce the needed computing power.
- I also train on only a small part of the data set.

4) validate the model
- The model does not seem to overfit and it may be that I should have trained it for more epochs. But this is a test project so I didn't want to spend too much time on the training.

5) save the model

Created by: Patrick Hallila 01/08/2022

"""

from re import U
from webbrowser import get
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential


DATA_PATH = "C://Users//Patrick//OneDrive - City, University of London//PhD//Programming projects//VSCode//NLP_projects//Sentiment_data//tripadvisor_hotel_reviews.csv"

def calculate_sentiment_label(rating):
    """
    I use the same logic as when calculating Net promotor scores:
    0 = Negative, 1 = Neutral, 2 = Positive

    """
    if rating < 4:
        return 0
    elif rating < 5:
        return 1
    return 2

def create_train_val_test_split(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def prepare_training_data(file_path, cut_data_size=False):
    
    #Download data
    df = pd.read_csv(file_path)

    #I cut the data size to reduce computing times
    if cut_data_size:
        df = df[:2000]
    
    # Convert the label to numerical
    df['performance'] = df.Rating.apply(calculate_sentiment_label)

    # Extract labels and features
    y = df.pop('performance')
    X = df.pop('Review')

    X_train, X_validation, X_test, y_train, y_validation, y_test = create_train_val_test_split(X,y)

    #Convert to tensors
    X_train = tf.convert_to_tensor(X_train)
    X_validation = tf.convert_to_tensor(X_validation)
    X_test = tf.convert_to_tensor(X_test)
    y_train = tf.convert_to_tensor(y_train)
    y_validation = tf.convert_to_tensor(y_validation)
    y_test = tf.convert_to_tensor(y_test)

    print("Data is sucessfully converted to tensors!")

    return X_train, X_validation, X_test, y_train, y_validation, y_test



def create_model(model_url, preprocessor_url):
    encoder = get_model(model_url)
    preprocessor = get_model(preprocessor_url)

    # Freeze the existing model
    encoder.trainable = False

    #Prepare preprocessor
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    encoder_inputs = preprocessor(text_input)

    #Choose outputs
    outputs = encoder(encoder_inputs)
    pooled_output = outputs["pooled_output"]  
    
    #Create embedding model
    embedding_model = tf.keras.Model(text_input, pooled_output)

    model = tf.keras.Sequential([
    embedding_model,
    tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.summary()

    model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['acc']
    )

    return model

def get_model(url):
    return hub.KerasLayer(url)

def evaluate_model(model, X_test, y_test):

    #Print key metrics
    print("Evaluate on test data")
    results = model.evaluate(X_test, y_test)
    print("test loss, test acc:", results)

    print("Predictions made by model:")
    for i in [35,78,123]:
        prediction = model.predict(X_test[i])
        if np.argmax(prediction) == 0:
            value = "Negative"
        elif np.argmax(prediction) == 1:
            value = "Neutral"
        else:
            value = "Positive"
        print(f"Message: {X_test[i]} --> {value}, {prediction[np.argmax(prediction)]}\n")


if __name__=="__main__":
    
    #Prepare data
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_training_data(DATA_PATH, cut_data_size=True)

    # Create model
    model = create_model(
        model_url="https://tfhub.dev/tensorflow/albert_en_base/3",
        preprocessor_url="https://tfhub.dev/tensorflow/albert_en_preprocess/3"
    )

    # Fit the model
    history = model.fit(X_train, y_train, epochs=5,validation_data=(X_validation, y_validation))

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    #Save model
    model.save('my_model.h5')