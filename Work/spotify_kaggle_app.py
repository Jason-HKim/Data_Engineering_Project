import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PART 1

st.write("# Spotify Data Explorer")

# PART 2: DATA IMPORT

    # NOTE: Because the csv files are large, to run the app, type: 
    # run spotify_kaggle_app.py --server.maxUploadSize=1028

tracks_df = pd.read_csv('data/tracks_df.csv')

artists_df = pd.read_csv('data/artists_df.csv')


    # Displaying random sample of Tracks & Artists Dataframes
st.write("## Random Snapshots of Tracks & Artists Dataframes:")

st.write('### Random Track Data Snapshot')
if st.button('Generate', key="tracks_data"):
    st.dataframe(tracks_df.sample(n=5000))
    st.write('5000 random rows loaded from tracks_df')

st.write('### Random Artists Data Snapshot')
if st.button('Generate', key="artists_data"):
    st.dataframe(artists_df.sample(n=5000))
    st.write('5000 random rows loaded from artists_df')






# plt.figure(facecolor='white')
# fig, ax = plt.subplots(figsize=(30,10))

# sns.histplot(data=artists_df, x=artists_df['followers'])

# plt.title('Distribution of No. Followers of All Artists')

# show_graph = st.checkbox('Show Graph', value=True)

# if show_graph:
#     st.pyplot(fig)


# PART 5: INPUT SONG/ARTIST NAME FOR METADATA:


st.write("## Song Lookup")
song = st.text_input("Song Name").lower()
song_artist = st.text_input("Song Artist").lower()
 
if st.button('Search', key = 'song'):
    if song and song_artist:
        st.write(tracks_df.loc[(tracks_df['name'].str.lower()==song) & tracks_df['artists'].str.lower().apply(lambda x: song_artist in x.lower())])
    elif song:
        st.write(tracks_df.loc[tracks_df['name'].str.lower()==song])
    elif song_artist:
        st.write(tracks_df.loc[tracks_df['artists'].str.lower().apply(lambda x: song_artist in x.lower())])



st.write("## Artist Lookup")

lookup_artist = st.text_input("Artist", value=None)

if st.button('Search', key="artist"):
    st.write(artists_df.loc[artists_df['name'].str.lower()==lookup_artist.lower()])

# st.write(tracks_df[tracks_df['artists'].apply(lambda x: artist in x)])


# PART 6 TRAINING A LINEAR REGRESSION: PREDICTING POPULARITY
st.write("## Predicting Popularity:")



# PART 7:



import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# can't import standard scaler, ridge, or polynomialfeatures from sklearn

# X = tracks_df[['duration_sec', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']]

# y = tracks_df['popularity']

# X, X_test, y, y_test = train_test_split(X, y, test_size=.2, random_state=10)

# X, y = np.array(X), np.array(y)

# kf = KFold(n_splits=5, shuffle=True, random_state = 42)

# cv_lm_reg_r2s = []

# for train_ind, val_ind in kf.split(X,y):
    
#     X_train, y_train = X[train_ind], y[train_ind]
#     X_val, y_val = X[val_ind], y[val_ind] 
    
#     lm_reg = Ridge(alpha=1)

#     #ridge with feature scaling
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.transform(X_val)
    
#     lm_reg.fit(X_train_scaled, y_train)
#     cv_lm_reg_r2s.append(lm_reg.score(X_val_scaled, y_val))


# st.write(f'Ridge scores: {cv_lm_reg_r2s} \n')
# st.write(f'Ridge mean cv r^2: {np.mean(cv_lm_reg_r2s):.3f} +- {np.std(cv_lm_reg_r2s):.3f}')

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_test_scaled = scaler.transform(X_test)

# lm_reg = Ridge(alpha=1)
# lm_reg.fit(X_scaled,y)
# print(f'Ridge Regression test R^2: {lm_reg.score(X_test_scaled, y_test):.3f}')


# X = tracks_df[['duration_sec', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']]

X = tracks_df[['duration_sec', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]

y = tracks_df['popularity']


# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.25, random_state=3)


# lm = LinearRegression()
# scaler = StandardScaler()

# X_train_scaled = scaler.fit_transform(X_train.values)
# X_val_scaled = scaler.transform(X_val.values)
# X_test_scaled = scaler.transform(X_test.values)

# lm.fit(X_train, y_train)
# print(f'Linear Regression val R^2: {lm.score(X_val, y_val):.3f}')

# lm.fit(X,y)
# print(f'Linear Regression test R^2: {lm.score(X_test, y_test):.3f}')



X_train, X_test, y_train, y_test = train_test_split(X,y)

lr = LinearRegression()

lr.fit(X_train, y_train)

# Graph:

st.write("Distribution of popularity level among all artists in the dataset:")
plt.figure(facecolor='white')
fig, ax = plt.subplots(figsize=(30,10))

sns.histplot(data=tracks_df, x=tracks_df['popularity'])

plt.title('Distribution of Popularity Level of All Songs')

show_graph = st.checkbox('Show Graph', value=True)

if show_graph:
    st.pyplot(fig)

# time_signature = st.number_input("Time Signature (0-5)", value=3)
tempo = st.number_input("Tempo (0-250)", value=200)
valence = st.number_input("Valence (0.0 - 1.0)", value = 0.8)
liveness = st.number_input("Liveness (0.0 - 1.0)", value=0.2)
instrumentalness = st.number_input("Instrumentalness (0.0 - 1.0)", value=0.4)
acousticness = st.number_input("Acousticness (0.0 - 1.0)", value=0.5)
speechiness = st.number_input("Speechiness (0.0-1.0)", value=0.5)
# mode = st.number_input("Mode (0 or 1)", value=0)
loudness = st.number_input("Loudness (-60-0 db)", value=-35)
# key = st.number_input("Key (0-10)", value=4)
energy = st.number_input("Energy (0.0-1.0)", value=0.6)
danceability = st.number_input("Danceability (0.0-1.0)", value=0.6)
# explicit = st.number_input("Explicit (0 = False, 1 = True)", value=0)
duration_sec = st.number_input("Duration of the Song (seconds)", value=240)


# input_data = pd.DataFrame({'time_signature':[time_signature], 'tempo':[tempo],'valence':[valence], 'liveness':[liveness], 'instrumentalness':[instrumentalness], 'acousticness':[acousticness],'speechiness':[speechiness], 'mode':[mode], 'loudness':[loudness], 'key':[key],'energy':[energy], 'danceability':[danceability], 'explicit':[explicit], 'duration_sec':[duration_sec]})

input_data = pd.DataFrame({'tempo':[tempo],'valence':[valence], 'liveness':[liveness], 'instrumentalness':[instrumentalness], 'acousticness':[acousticness],'speechiness':[speechiness], 'loudness':[loudness],'energy':[energy], 'danceability':[danceability], 'duration_sec':[duration_sec]})


pred = lr.predict(input_data)[0]

if st.button('Calculate', key='popularity_predict'):
   st.write(f'Predicted Song Popularity: {pred}')
   st.write(f'R-squared of model: {lr.score(X_test, y_test):.2f}')
