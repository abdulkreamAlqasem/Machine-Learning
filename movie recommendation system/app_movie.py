import streamlit as st
import pickle
import numpy as np
import pandas as pd
from surprise import Dataset,Reader,SVD
from surprise.model_selection import cross_validate
from collections import defaultdict


with open("saved_steps.pkl","rb") as file:
    data=pickle.load(file)


my_model=data["model"]
my_final_dataset_org=data["final_data_org"]
final_dataset_norg=data["final_data_norg"]
final_dataset_for_link=data["final_dataset_for_link"]
predictions=data["predictions"]


st.title("movie recommendation system")
st.write("## the user enter the number user id he want then he give him his top N recommendation system ")

x=st.slider("select a number of id user from 1 to 610",1,610)
st.write("you selected ",x)

st.write("### user id of number {} and his rating list ".format(x))
st.write(my_final_dataset_org[my_final_dataset_org["userId"] == x])


reader = Reader(rating_scale=(1, 5))
data_set = Dataset.load_from_df(final_dataset_norg, reader)

train_set = data_set.build_full_trainset()
my_model.fit(train_set)

testset = train_set.build_testset()
predictions = my_model.test(testset)

#this function i take from https://surprise.readthedocs.io/en/stable/FAQ.html
#from surprise lib to get top 5 from same lib
def get_top_n(predictions, n=5):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 5.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def top_n_recs(user_id, top_n):
    top_n = get_top_n(predictions, n=top_n)
    return pd.DataFrame(top_n[user_id], columns=["movieId", "rating"])

top_N=st.slider("select the number of N in top N to see",1,20)

st.write("### \nTop %d recommendations for user %d" % (top_N, x))

st.write("--------------------------")

top_5_df=top_n_recs(x,top_N)
for i in range(top_N):
    movie=final_dataset_for_link.loc[top_5_df["movieId"].loc[i]]
    st.write("title is : ",movie["title"])
    st.write("genres : ",movie["genres"])
    st.write("the {title} in TMDP is : https://www.themoviedb.org/movie/{number_of_movie}".format(title=movie["title"],number_of_movie=int(movie["tmdbId"])))
    st.write("--------------------------")


