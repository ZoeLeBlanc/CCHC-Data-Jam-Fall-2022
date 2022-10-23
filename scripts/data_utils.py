import pandas as pd
import os
from PIL import Image
from tqdm import tqdm

def get_local_image_locations(dir, output_path):
    """Function to get the image locations on my local computer
    :param dir: path to the data folder
    :param output_path: path to the output folder
    :return: pandas dataframe"""
    if os.path.exists(output_path):
        local_files = pd.read_csv(output_path)
    else:
        dfs = []
        for r, d, f in os.walk(dir):
            for file in f:
                if file.endswith(".jpg"):
                    dfs.append(pd.DataFrame([{'file_id': file, 'location': r}]))
        local_files = pd.concat(dfs)
        local_files.to_csv(output_path, index=False)
    return local_files

def get_initial_dataset(output_path, is_sample):
    """Function to load data from csv files and return a pandas dataframe
    :param output_path: path to the output folder
    :return: pandas dataframe"""
    if os.path.exists(output_path):
        merged_df = pd.read_csv(output_path)
    else:
        metadata_path = 'private_materials/sample-data/metadata.csv' if is_sample else 'private_materials/metadata.csv'
        metadata_df = pd.read_csv(metadata_path)
        manifest_path = 'private_materials/sample-data/manifest.txt' if is_sample else 'private_materials/manifest.txt'
        manifest_df = pd.read_csv(manifest_path, low_memory=False, header=None, sep='\t', names=['file_id', 'MD5_hash', 'data_location'])
        manifest_df[['split_id', 'image_filename']]= manifest_df['file_id'].str.split('_', expand=True)
        metadata_df['split_id'] = metadata_df['id'].str.split('/').str[4]
        merged_df = pd.merge(metadata_df, manifest_df, on='split_id')
        merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('^Unnamed')]
        merged_df.to_csv(output_path, index=False)
    return merged_df

def get_image_sizes(df):
    """Function to get the image sizes from the data
    :param df: pandas dataframe
    :return: pandas dataframe"""
    df['image_error'] = False
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Getting image sizes'):
        try:
            image = Image.open('private_materials/data/' + row['file_id'] + '.jpg')
            df.loc[index, 'width'] = image.width
            df.loc[index, 'height'] = image.height
        except:
            df.loc[index, 'image_error'] = True
            df.loc[index, 'width'] = None
            df.loc[index, 'height'] = None
    return df