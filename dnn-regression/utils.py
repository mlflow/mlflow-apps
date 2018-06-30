import os
import re
import sys
import pandas

# Downloads the diamonds dataset into the provided folder path.
# Converts categorical values into numerical values and splits the data into training/testing sets.
def download_diamonds(temp_folder_path):

    #Downloading csv file from ggplot2's hosted dataset on github.
    url = "https://raw.githubusercontent.com/tidyverse/ggplot2/4c678917/data-raw/diamonds.csv"
    print("Downloading diamonds csv file to %s..." % temp_folder_path)

    pd_df = pandas.read_csv(url)

    # Conversion of qualitative values to quantitative values. For diamonds only.
    pd_df['cut'] = pd_df['cut'].replace({'Fair':0, 'Good':1, 
                                                    'Very Good':2, 'Premium':3, 'Ideal':4})
    pd_df['color'] = pd_df['color'].replace({'J':0, 'I':1, 
                                                        'H':2, 'G':3, 'F':4, 'E':5, 'D':6})
    pd_df['clarity'] = pd_df['clarity'].replace({'I1':0, 
                            'SI1':1, 'SI2':2, 'VS1':3, 'VS2':4, 'VVS1':5, 'VVS2':6, 'IF':7})

    pd_df.rename(columns=lambda x: re.sub(r'[^\w]', '', x), inplace=True)
    # Splitting the data up so that 80% of the data is training data, 20% testing data.
    training_data = pd_df[:int(pd_df.shape[0]*.8)]
    testing_data = pd_df[int(pd_df.shape[0]*.8):]
    print("Downloaded diamonds csv file.")
    print("Creating diamonds dataset parquet files...")
    with open(os.path.join(temp_folder_path, "train_diamonds.parquet"), 'w'):
        training_data.to_parquet(os.path.join(temp_folder_path, "train_diamonds.parquet"))
    with open(os.path.join(temp_folder_path, "test_diamonds.parquet"), 'w'):
        testing_data.to_parquet(os.path.join(temp_folder_path, "test_diamonds.parquet"))
    print("Diamonds dataset parquet files created.")

    return pd_df

def get_feature_cols(feat_cols, label_col, all_cols):
    # This is the case if the user specified which columns are to be feature columns.
    if feat_cols:
        return feat_cols
    # If no feature columns are specified, it is assumed all columns but the label are features.
    else:
        feats = []
        for col in all_cols:
            if col != label_col:
                feats.append(col)
        return feats

if __name__ == '__main__':
    download_diamonds(sys.argv[1])
