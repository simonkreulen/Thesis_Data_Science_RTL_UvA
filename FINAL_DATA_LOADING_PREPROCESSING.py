# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ### In this notebook, the data is loaded into pandas dataframes, and all initial preprocessing steps are done. Because of company regulations, the RTL data and file names (Videoland) are excluded.

# COMMAND ----------

import numpy as np
import pandas as pd

# COMMAND ----------

# loading in the open source MPST dataset after downloading the CSV from Kaggle.

mpst_df = pd.read_csv('mpst_full_data.csv')
mpst_df.head()
len(mpst_df)

# COMMAND ----------

# nulls and duplicates check:

columns_to_check = [
    'imdb_id', 'title', 'plot_synopsis', 'tags', 'split', 'synopsis_source'
]

for col in columns_to_check:
    unique_count = mpst_df[col].nunique()
    print(f"Total unique entries in {col}: {unique_count}")

for col in columns_to_check:
    nulls_count = mpst_df[col].isna().sum()
    print(f"Total null entries in {col}: {nulls_count}")

# COMMAND ----------

# Print the collection of unique tags used:

unique_tags = set(tag.strip() for tags in mpst_df['tags'] for tag in tags.split(','))

# Convert to list or print the set directly
print(unique_tags)
len(unique_tags)

# COMMAND ----------

empty_tags_count = mpst_df['tags'].apply(lambda x: x == '' or pd.isna(x)).sum()

print(f"Empty cells in 'tags' column: {empty_tags_count}")

# COMMAND ----------

mpst_df = mpst_df[(mpst_df['tags'].notna()) & (mpst_df['tags'] != '')]
len(mpst_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Querying Wikidata for Awards data

# COMMAND ----------

#creating chunks of imdb_ids from MPST data sized 500:

mpst_df['imdb_id'].to_csv("imdb_ids_mpst", index=False, header=False)

# COMMAND ----------

with open("imdb_ids_mpst", "r") as file:
    imdb_ids = [line.strip() for line in file.readlines()]

chunk_size = 500

# split the IMDb IDs into chunks of 500
imdb_chunks = [imdb_ids[i:i + chunk_size] for i in range(0, len(imdb_ids), chunk_size)]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Use the following query in Wikidata Query Service, and fill in the imdb_ids from the generated chunk files to retrieve the award data:

# COMMAND ----------

# MAGIC %md
# MAGIC Link: https://query.wikidata.org/
# MAGIC

# COMMAND ----------

# %sql

#     SELECT ?imdb_id ?title ?award_label WHERE {
#       ?film wdt:P31 wd:Q11424;  # P31 = instance of, Q11424 = film
#             wdt:P345 ?imdb_id;   # P345 = IMDb ID
#             wdt:P166 ?award.     # P166 = award received
#       ?film rdfs:label ?title.
#       ?award rdfs:label ?award_label.
#       FILTER(?imdb_id IN ("....."))
#       FILTER(LANG(?title) = "en")
#       FILTER(LANG(?award_label) = "en")
#     }

# COMMAND ----------

# MAGIC %md
# MAGIC #### Download the response files from Wikidata Query Service, and upload to the notebooks.

# COMMAND ----------

wiki_awards = pd.read_csv('merged_awards.csv')
wiki_awards

# COMMAND ----------

wiki_awards = wiki_awards.groupby(['imdb_id', 'title'], as_index=False)['award_label'].apply(lambda x: ', '.join(x)).reset_index(drop=True)
wiki_awards = wiki_awards.drop(index=0).reset_index(drop=True)
wiki_awards

# COMMAND ----------


#spliting each award list by commas and unpack the values
all_awards = wiki_awards['award_label'].str.split(',').explode().str.strip()

#get the unique awards
unique_awards = all_awards.unique()

#sort the awards (optional)
unique_awards_sorted = sorted(unique_awards)

wiki_awards_names = []

for award in unique_awards_sorted:
    wiki_awards_names.append(award)

wiki_awards_names
len(wiki_awards_names)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Now that we have the wiki_awards, we merge them into the MPST dataset, and for titles with no prize, we fill in: No award, since this can also hold generative / predictive information.

# COMMAND ----------

merged_df = pd.merge(mpst_df, wiki_awards, on='imdb_id', how='left')

merged_df['award_label'].fillna('No award', inplace=True)

merged_df = merged_df[['imdb_id', 'title_x', 'plot_synopsis', 'tags', 'split', 'synopsis_source', 'award_label']]

merged_df.rename(columns={'title_x': 'title'}, inplace=True)

# COMMAND ----------

#split each entry in 'tags' column by comma and expand into a flat list
unique_tags = set(tag.strip() for tags in merged_df['tags'] for tag in tags.split(','))

print(unique_tags)

# COMMAND ----------

merged_df['tags'] = merged_df['tags'].str.split(',')

# COMMAND ----------

merged_df.head()

# COMMAND ----------

merged_df.to_csv("/dbfs/tmp/train_test_df.csv", index=False)

# COMMAND ----------

merged_df