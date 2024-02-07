import sys
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras import layers
import tensorflow as tf
import numpy as np
import pandas as pd

def main():
    
    train_df = pd.read_csv("train.csv")
    
    text_df = train_df[[
        'Name', 
        'Artist', 
        'Album Artist', 
        'Composer', 
        'Album', 
        'Genre'
    ]].astype(str)
    num_df = train_df[[
        'Total Time', 
        'Size', 
        'Track Number', 
        'Track Count', 
        'Year', 
        'Date Added', 
        'Date Modified', 
        'Release Date', 
        'Normalization', 
        'Play Count', 
        'Skip Count'
    ]].astype(int)
    label_df = train_df['Label'].astype(int)
    
    #parameters
    max_features = 30000
    embedding_dim = 4
    sequence_length = 11
    
    vectorize_layer = keras.layers.TextVectorization(
        standardize='lower_and_strip_punctuation',
        split='whitespace',
        max_tokens=max_features-1,
        output_mode="int",
        output_sequence_length=sequence_length,
    )
    text = text_df.apply(lambda x: ' '.join(x.dropna().values.tolist()), axis=1)
    vectorize_layer.adapt(text)

    ###########################################################################
    ## INPUTS
    ###########################################################################
    name = keras.layers.Input(shape=(1, ), dtype=str)
    vname = vectorize_layer(name)
    ename = layers.Embedding(max_features, embedding_dim)(vname)
    
    artist = keras.layers.Input(shape=(1, ), dtype=str)
    vartist = vectorize_layer(artist)
    eartist = layers.Embedding(max_features, embedding_dim)(vartist)
    
    album_artist = keras.layers.Input(shape=(1, ), dtype=str)
    valbum_artist = vectorize_layer(album_artist)
    ealbum_artist = layers.Embedding(max_features, embedding_dim)(valbum_artist)
                                    
    composer = keras.layers.Input(shape=(1, ), dtype=str)
    vcomposer = vectorize_layer(composer)
    ecomposer = layers.Embedding(max_features, embedding_dim)(vcomposer)
    
    album = keras.layers.Input(shape=(1, ), dtype=str)
    valbum = vectorize_layer(album)
    ealbum = layers.Embedding(max_features, embedding_dim)(valbum)
    
    genre = keras.layers.Input(shape=(1, ), dtype=str)
    vgenre = vectorize_layer(genre)
    egenre = layers.Embedding(max_features, embedding_dim)(vgenre)
    
    num_input = keras.layers.Input(shape=(11, ), dtype=int)
    enum = keras.layers.Embedding(max_features, embedding_dim)(num_input)
    
    ###########################################################################
    ## MODEL
    ###########################################################################
    merger = keras.layers.Concatenate()([ename, eartist, ealbum_artist, ecomposer, ealbum, egenre, enum])
    dense1 = layers.Dense(16, activation="sigmoid")(merger)
    dropout = layers.Dropout(0.5)(dense1)
    pool = layers.GlobalMaxPooling1D()(dropout)
    output = keras.layers.Dense(1, activation="sigmoid")(pool)
    
    model = keras.models.Model(inputs=[name, artist, album_artist, composer, album, genre, num_input], outputs=output)
    
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    model.fit(
        x=[
            train_df['Name'].astype(str), 
            train_df['Artist'].astype(str), 
            train_df['Album Artist'].astype(str), 
            train_df['Composer'].astype(str), 
            train_df['Album'].astype(str), 
            train_df['Genre'].astype(str), 
            train_df[[
                'Total Time',
                'Size',
                'Track Number',
                'Track Count',
                'Year',
                'Date Added',
                'Date Modified',
                'Release Date',
                'Normalization',
                'Play Count',
                'Skip Count'
            ]].astype(int)
        ], 
        y=train_df['Label'].astype(int), 
        shuffle=True,
        epochs=30
    )
    
    test_df = pd.read_csv('test.csv')
    prediction = model.predict([
        test_df['Name'].astype(str), 
        test_df['Artist'].astype(str), 
        test_df['Album Artist'].astype(str), 
        test_df['Composer'].astype(str), 
        test_df['Album'].astype(str), 
        test_df['Genre'].astype(str), 
        test_df[[
            'Total Time',
            'Size',
            'Track Number',
            'Track Count',
            'Year',
            'Date Added',
            'Date Modified',
            'Release Date',
            'Normalization',
            'Play Count',
            'Skip Count'
        ]].astype(int)
    ], 0)
    test_df["Confidence"] = prediction
    test_df.to_csv("prediction.csv", index=False)
    return
    
if __name__ == "__main__":
    main()