import pandas as pd
import os

def get_initial_dataset(output_path):
    """Function to load data from csv files and return a pandas dataframe
    :param output_path: path to the output folder
    :return: pandas dataframe"""
    if os.path.exists(output_path):
        merged_df = pd.read_csv(output_path)
    else:
        metadata_df = pd.read_csv('private_materials/metadata.csv', low_memory=False)
        manifest_df = pd.read_csv('private_materials/manifest.txt', low_memory=False, header=None, sep='\t', names=['file_id', 'MD5_hash', 'data_location'])
        manifest_df[['split_id', 'image_filename']]= manifest_df['file_id'].str.split('_', expand=True)
        metadata_df['split_id'] = metadata_df['id'].str.split('/').str[4]
        merged_df = pd.merge(metadata_df, manifest_df, on='split_id')
        merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('^Unnamed')]
        merged_df.to_csv('private_materials/datajam_data/initial_cleaned_dataset.csv', index=False)
    return merged_df