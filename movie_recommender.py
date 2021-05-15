# %% [code]
import numpy as np # linear algebra
import pandas as pd # data processing

# %% [code]
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %% [code]
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# %% [code]

def get_title_from_index(index):
  return df.loc[index, "movie_title"]

def get_index_from_title(movie_title):
  return df.loc[df.movie_title == movie_title].index[0]


# %% [code]
df = pd.read_csv("../input/final-data1csv/final_data1.csv")


# %% [code]
features = ['genres','director_name','actor_1_name','actor_2_name']


# %% [code]
for feature in features:
    df[feature] = df[feature].fillna('')

# %% [code]
def combine_features(row):
        return row['genres']+" "+row['director_name']+" "+row['actor_1_name']+" "+row['actor_2_name']
    

# %% [code]
df["combined_features"] = df.apply(combine_features,axis=1)
               
print("Combined Features:",df["combined_features"].head())

# %% [code]
cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])

# %% [code]
cosine_sim = cosine_similarity(count_matrix)
movie_user_likes = "avatar"

# %% [code]
movie_index = get_index_from_title(movie_user_likes)

# %% [code]
similar_movies = list(enumerate(cosine_sim[movie_index]))

# %% [code]
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)

# %% [code]
i=0
for element in sorted_similar_movies:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>70:
        break