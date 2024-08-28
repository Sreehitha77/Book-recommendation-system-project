import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# Load the data
books = pd.read_csv("Books.csv", on_bad_lines='skip', encoding='utf-8')
ratings = pd.read_csv('Ratings (1).csv')
users = pd.read_csv('Users (1).csv', on_bad_lines='skip', encoding='latin')

# Merge the datasets
book_ratings = pd.merge(left=books, right=ratings, how='left', left_on='ISBN', right_on='ISBN')
book_ratings_users = pd.merge(left=book_ratings, right=users, how='right', left_on='User-ID', right_on='User-ID')

# Drop NA values
book_ratings_users = book_ratings_users.dropna()

# Streamlit app
st.title('Book Recommendation System')

# Sidebar for user inputs
st.sidebar.header('User Input Features')
selected_rating = st.sidebar.slider('Book Rating', 0, 10, (0, 10))
selected_age = st.sidebar.slider('User Age', 0, 100, (0, 100))

# Filter data based on user input
filtered_data = book_ratings_users[
    (book_ratings_users['Book-Rating'] >= selected_rating[0]) &
    (book_ratings_users['Book-Rating'] <= selected_rating[1]) &
    (book_ratings_users['Age'] >= selected_age[0]) &
    (book_ratings_users['Age'] <= selected_age[1])
]

st.subheader('Filtered Data')
st.write(filtered_data.head())

# Display popular books
st.subheader('Popular Books')
popular_books = filtered_data['Book-Title'].value_counts().head(10)
st.bar_chart(popular_books)

# Display book ratings distribution
st.subheader('Book Ratings Distribution')
ratings_distribution = filtered_data['Book-Rating'].value_counts().sort_index()
st.bar_chart(ratings_distribution)

# Clustering
st.subheader('Clustering')
cluster_method = st.selectbox('Select Clustering Method', ('KMeans', 'DBSCAN', 'AgglomerativeClustering'))

if cluster_method == 'KMeans':
    n_clusters = st.slider('Number of Clusters (KMeans)', 2, 10, 3)
    model = KMeans(n_clusters=n_clusters)
elif cluster_method == 'DBSCAN':
    eps = st.slider('Epsilon (DBSCAN)', 0.1, 10.0, 0.5)
    min_samples = st.slider('Min Samples (DBSCAN)', 1, 10, 5)
    model = DBSCAN(eps=eps, min_samples=min_samples)
else:
    n_clusters = st.slider('Number of Clusters (Agglomerative)', 2, 10, 3)
    model = AgglomerativeClustering(n_clusters=n_clusters)

features = filtered_data[['Age', 'Book-Rating']].dropna()
model.fit(features)
filtered_data['Cluster'] = model.labels_

st.subheader('Clustered Data')
st.write(filtered_data[['Book-Title', 'Age', 'Book-Rating', 'Cluster']].head())

# Visualization of clusters
st.subheader('Clusters Visualization')
fig, ax = plt.subplots()
scatter = ax.scatter(filtered_data['Age'], filtered_data['Book-Rating'], c=filtered_data['Cluster'], cmap='viridis')
legend = ax.legend(*scatter.legend_elements(), title='Clusters')
ax.add_artist(legend)
plt.xlabel('Age')
plt.ylabel('Book Rating')
st.pyplot(fig)

if __name__ == "__main__":
    st.run()
