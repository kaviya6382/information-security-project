# Paste these cells between the 3 starter cells and the final test cell
# of the freeCodeCamp notebook. The starter cells already define:
# df_books (isbn, title) and df_ratings (user, isbn, rating), plus the imports
# for numpy, pandas, csr_matrix and NearestNeighbors.


# ======================================================================
# Cell A: clean the data
# ======================================================================
# keep users with at least 200 ratings
user_counts = df_ratings['user'].value_counts()
df = df_ratings[df_ratings['user'].isin(user_counts[user_counts >= 200].index)]

# attach titles, then keep books with at least 100 ratings
df = df.merge(df_books, on='isbn')
df = df.drop_duplicates(['user', 'title'])        # same user rating the same title twice
book_counts = df['title'].value_counts()
df = df[df['title'].isin(book_counts[book_counts >= 100].index)]


# ======================================================================
# Cell B: build the title x user matrix and fit the model
# ======================================================================
pivot = df.pivot(index='title', columns='user', values='rating').fillna(0)
matrix = csr_matrix(pivot.values)

model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(matrix)


# ======================================================================
# Cell C: the recommendation function
# ======================================================================
def get_recommends(book=""):
    idx = pivot.index.get_loc(book)
    # ask for 6 neighbours: the closest one is the book itself
    distances, indices = model.kneighbors(matrix[idx], n_neighbors=6)

    recommended = [
        [pivot.index[i], float(d)]
        for i, d in zip(indices[0][1:], distances[0][1:])
    ]
    recommended = recommended[::-1]    # expected output lists the farthest first
    return [book, recommended]


get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
