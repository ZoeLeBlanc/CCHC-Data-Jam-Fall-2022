import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
from ast import literal_eval
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import re
from thefuzz import fuzz
import itertools

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
    rows_without_images = df[df.width.isna()]
    rows_with_images = df[~df.width.isna()]
    if len(rows_without_images) > 0:
        rows_without_images['image_error'] = False
        for index, row in tqdm(rows_without_images.iterrows(), total=rows_without_images.shape[0], desc='Getting image sizes'):
            try:
                image = Image.open('private_materials/data/' + row['file_id'] + '.jpg')
                rows_without_images.loc[index, 'width'] = image.width
                rows_without_images.loc[index, 'height'] = image.height
            except:
                rows_without_images.loc[index, 'image_error'] = True
                rows_without_images.loc[index, 'width'] = None
                rows_without_images.loc[index, 'height'] = None
        df = pd.concat([rows_with_images, rows_without_images])
    else:
        df = rows_with_images
    return df

# dictionary containing all months and abbreviations
months = {'January': '01', 'February': '02', 'March': '03', 'April': '04', 'May': '05', 'June': '06', 'July': '07', 'August': '08', 'September': '09', 'October': '10', 'November': '11', 'December': '12'}

def clean_item_dates(row):
    year = None
    month = None
    day = None
    
    if pd.isnull(row['item_date']) == False:
        if len(row.date) < 10:
            split_date = row['item_date'].replace('[', '').replace(']', '').replace('-', ' ').split(' ')
            for item in split_date:
                striped_item = re.sub(r'[^\w\s]','',item)
                if any(i.isdigit() for i in striped_item):
                    if len(striped_item) > 3:
                        year = re.findall(r'\d{4}', striped_item)
                        if len(year) > 0:
                            year = row.date  if len(row.date) == 4 else year[0]
                    else:
                        day = re.findall(r'\d+', striped_item)
                        if len(day) > 0:
                            if len(day[0]) == 2:
                                day = day[0]
                            elif len(day[0]) == 1:
                                day = '0' + day[0]
                            elif len(day[0]) > 2:
                                day = '01'
                else:
                    
                    for key in months:
                        if striped_item in key:
                            month = months[key]
        else:
            year, month, day = row.date.split('-')
    return year, month, day

def finalize_dates(dates_df):
    for index, row in tqdm(dates_df.iterrows(), desc="Cleaning dates", total=len(dates_df)):
        if pd.isnull(row['year']):
            if pd.isnull(row['min_date']) == False:
                dates_df.loc[index, 'year'] = row['min_date'].year
            elif row['date'] != None:
                dates_df.loc[index, 'year'] = row['date']
        if pd.isnull(row['month']):
            dates_df.loc[index, 'month'] = row['min_date'].strftime('%m') if pd.isnull(row['min_date']) == False else '01'
        if pd.isnull(row['day']):
            dates_df.loc[index, 'day'] = row['min_date'].strftime('%d') if pd.isnull(row['min_date']) == False else '01'
    return dates_df

def check_if_file_exists(output_path, column_name):
    """Function to check if the file exists
    :param output_path: path to the output folder
    :return: boolean"""
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        if column_name in df.columns:
            load_file = True
        else:
            load_file = False
    return load_file


def get_dates(merged_df, output_path):
    if 'cleaned_date' not in merged_df.columns:
        load_file = check_if_file_exists(output_path, 'cleaned_date')
        if load_file == False:
            merged_df['dates'] = merged_df.dates.fillna('[]')
            merged_df['dates'] = merged_df.dates.apply(literal_eval)
            date_cols = ['date',
                'item_created_published',
                'item_created_published_date',
                'item_date',
                'item_source_created',
                'dates',
                'item_source_created',
                'item_sort_date']
            
            merged_df.date = merged_df.date.astype(str)
            dates_df = merged_df[['id'] + date_cols]
            tqdm.pandas(desc="Cleaning item dates")
            dates_df['year'], dates_df['month'], dates_df['day'] = zip(*merged_df.progress_apply(clean_item_dates, axis=1))
            exploded_dates = dates_df[['id', 'dates', 'date', 'item_date', 'month', 'year', 'day']].explode('dates')
            exploded_dates['str_dates'] = exploded_dates.dates.str.split('T').str[0]
            exploded_dates['str_dates'] = pd.to_datetime(exploded_dates['str_dates'], errors='coerce', format="%Y-%m-%d")
            subset_dates = exploded_dates.groupby('id').agg({'str_dates': np.min }).reset_index().rename(columns={'str_dates':'min_date'})
            dates_df = dates_df.merge(subset_dates, on='id', how='left')
            dates_df = finalize_dates(dates_df)
            dates_df.year = dates_df['year'].astype(str)
            dates_df.loc[dates_df.year == '[]', 'year'] = dates_df.item_sort_date.astype(str)
            dates_df.loc[dates_df.item_date == 'c1903 July18.', 'month'] = months['July']
            dates_df.loc[dates_df.item_date == 'c1903 July18.', 'day'] = '18'
            dates_df.loc[dates_df.year == '1092', 'year'] = '1902'
            dates_df.loc[dates_df.year.str.contains('.'), 'year'] = dates_df.year.str.split('.').str[0]
            dates_df['final_date'] = dates_df['year'].astype(str) + '-' + dates_df['month'].astype(str) + '-' + dates_df['day'].astype(str)
            dates_df['cleaned_date'] = pd.to_datetime(dates_df['final_date'], errors='coerce', format="%Y-%m-%d")
            merged_df = merged_df.merge(dates_df[['id', 'cleaned_date', 'final_date']], on='id', how='left')
            merged_df.to_csv(output_path, index=False)
        else:
            merged_df = pd.read_csv(output_path)
    return merged_df

def get_matches(combinations):
    matches = []
    for c in tqdm(combinations, desc="Getting matches", total=len(combinations)):
        ratio = fuzz.ratio(c[0], c[1])
        if ratio > 65:
            split_c0 = c[0].split(' ')
            split_c1 = c[1].split(' ')
            total_check = 0
            for term in split_c0:
                if term in split_c1:
                    total_check += 1
            if total_check > 1:
                matches.append(pd.DataFrame([{'official_contributor':c[0], 'alternative_contributor':c[1], 'ratio':ratio}]))
    matches_df = pd.concat(matches)
    return matches_df

def clean_contributors(contributors_df, matches_df):
    contributors_df['cleaned_contributor'] = None
    for index, row in tqdm(contributors_df.iterrows(), desc="Cleaning contributors", total=len(contributors_df)):

        alternative_rows = matches_df[matches_df.alternative_contributor == row.stripped_contributor]
        original_rows = matches_df[matches_df.official_contributor == row.stripped_contributor]
        all_rows = pd.concat([alternative_rows, original_rows])
        if len(all_rows) > 0:
            all_rows = all_rows.sort_values(by='ratio', ascending=False)
            top_candidate = all_rows[0:1].official_contributor.values[0]
            contributors_df.loc[index, 'cleaned_contributor'] = contributors_df[contributors_df.stripped_contributor == top_candidate].contributor.values[0]
        else:
            contributors_df.loc[index, 'cleaned_contributor'] = row.contributor
    return contributors_df

def get_contributors(merged_df, output_path):
    if 'cleaned_contributor' not in merged_df.columns:
        load_file = check_if_file_exists(output_path, 'cleaned_contributor')
        if load_file == False:
            merged_df.contributor = merged_df.contributor.fillna('[]')
            merged_df.contributor = merged_df.contributor.apply(literal_eval)
            contributors_df = merged_df[['id', 'contributor']]
            contributors_df = contributors_df.explode('contributor')
            contributors_df = contributors_df[~contributors_df.contributor.isna()]
            contributors_df['stripped_contributor'] = contributors_df.contributor.str.replace('[^\w\s]','').str.lower()
            contributors = contributors_df.stripped_contributor.unique().tolist()
            combinations = itertools.combinations(contributors, 2)
            matches_df = get_matches(list(combinations))
            contributors_df = clean_contributors(contributors_df, matches_df)
            cleaned_contributors = contributors_df[['id', 'cleaned_contributor']].drop_duplicates()
            merged_df = merged_df.merge(cleaned_contributors, on='id', how='left')
            merged_df.to_csv(output_path, index=False)
        else:
            merged_df = pd.read_csv(output_path)
    return merged_df
   