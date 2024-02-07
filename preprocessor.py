import sys
import pandas as pd

def main():
    if len(sys.argv) < 3:
        return
   
    library_df = read_csv(sys.argv[1])
    seed_df = read_csv(sys.argv[2])
    print(len(library_df))
    print(len(seed_df))
    
    train_df, test_df = build_train_test_df(library_df, seed_df)
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)
    print(len(train_df))
    print(len(test_df))
    
    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    return

def read_csv(filename):
    dataframe = pd.read_csv(filename)
    dataframe = dataframe[[
        'Persistent ID',
        'Name',             # string -> string
        'Artist',           # string -> string
        'Album Artist',     # string -> string
        'Composer',         # string -> string
        'Album',            # string -> string
        'Genre',            # string -> string
        'Total Time',       # string -> int
        'Size',             # string -> int
        'Track Number',     # string -> int
        'Track Count',      # string -> int
        'Year',             # string -> int
        'Date Added',       # string -> int
        'Date Modified',    # string -> int
        'Release Date',     # string -> int
        'Normalization',    # string -> int
        'Play Count',       # string -> int
        'Skip Count'        # string -> int
    ]]
    return dataframe

def build_train_test_df(library, playlist):
    mask = library["Persistent ID"].isin(playlist["Persistent ID"])
    
    # provided playlist seed, labelled 1
    seed_df = library[mask]
    seed_df['Label'] = 1
    
    # randomly select songs in library but not playlist, labelled 0
    rand_df = library[~mask].sample(n=500)
    rand_df['Label'] = 0
    
    # concatenate seed and random sample to create training set
    train_df = pd.concat([seed_df, rand_df], axis=0)
    
    # test set is the rest of the library
    mask = library["Persistent ID"].isin(train_df["Persistent ID"])
    test_df = library[~mask]
    
    return train_df, test_df

def preprocess(dataframe):
    # fill missing information
    dataframe['Name'] = dataframe['Name'].fillna('')
    dataframe['Artist'] = dataframe['Artist'].fillna('')
    dataframe['Album Artist'] = dataframe['Album Artist'].fillna('')
    dataframe['Composer'] = dataframe['Composer'].fillna('')
    dataframe['Album'] = dataframe['Album'].fillna('')
    dataframe['Genre'] = dataframe['Genre'].fillna('')
    
    # convert dates
    dataframe['Release Date'] = dataframe['Release Date'].fillna('0000').str.slice(0,4).astype(int)
    dataframe['Date Added'] = dataframe['Date Added'].fillna('0000').str.slice(0,4).astype(int)
    dataframe['Date Modified'] = dataframe['Date Modified'].fillna('0000').str.slice(0,4).astype(int)
    
    # convert floats
    dataframe['Total Time'] = dataframe['Total Time'].fillna(0).astype(int)
    dataframe['Size'] = dataframe['Size'].fillna(0).astype(int)
    dataframe['Track Number'] = dataframe['Track Number'].fillna(0).astype(int)
    dataframe['Track Count'] = dataframe['Track Count'].fillna(0).astype(int)
    dataframe['Year'] = dataframe['Year'].fillna(0).astype(int)
    dataframe['Normalization'] = dataframe['Normalization'].fillna(0).astype(int)
    dataframe['Play Count'] = dataframe['Play Count'].fillna(0).astype(int)
    dataframe['Skip Count'] = dataframe['Skip Count'].fillna(0).astype(int)
    
    return dataframe

if __name__ == "__main__":
    main()