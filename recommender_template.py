import numpy as np
import pandas as pd
import recommender_functions as rf
import sys # can use sys to take command line arguments

class Recommender():
    '''
    This class uses FunkSVD to make predictions of movie ratings. Also, if its unable to use FunkSVD, it will use
    Knowledge Based recommendation (highest ranked) to make recommendations for users. If a movie is given, then it will
    use a Content Based Recommendation approach.
    '''
    def __init__(self, ):
        '''
        what do we need to start out our recommender system
        '''



    def fit(self, reviews_loc, movies_loc, latent_features=15, n_iter=100, learning_rate=0.001 ):
        '''
        fit the recommender to your dataset and also have this save the results
        to pull from when you need to make predictions

        :param reviews_loc: path to the reviews dataset (str)
        :param movies_loc: path to the movies dataset (str)
        :param latent_features: number of latent features to keep (int)
        :param n_iter: number of iterations (int)
        :param learning_rate: the learning rate (float)

        :returns None
        '''

        # Read in the datasets
        self.movies = pd.read_csv(movies_loc)
        self.reviews = pd.read_csv(reviews_loc)

        del self.movies['Unnamed: 0']
        del self.reviews['Unnamed: 0']

        # Create user-by-item matrix
        self.train_df = self.reviews[['user_id', 'movie_id', 'rating', 'timestamp']]
        self.user_item_df = self.train_df.groupby(['user_id', 'movie_id'])['rating'].max().unstack()

        self.user_item_matrix = np.array(self.user_item_df)
        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iter = n_iter

        # Set up useful values to be used through the rest of the function
        self.n_users = self.user_item_matrix.shape[0]
        self.n_movies = self.user_item_matrix.shape[1]
        self.n_ratings = np.count_nonzero(~np.isnan(self.user_item_matrix))
        self.movie_ids = np.array(self.user_item_df.columns)
        self.user_ids = np.array(self.user_item_df.index)

        # initialize the user and movie matrices with random values
        user_mat = np.random.rand(self.n_users, self.latent_features)
        movie_mat = np.random.rand(self.latent_features, self.n_movies)

        # initialize sse at 0 for first iteration
        sse_accum = 0

        # header for running results
        print("Optimization Statistics")
        print("Iterations | Mean Squared Error ")

        # for each iteration
        for i in range(n_iter):
            # update our sse
            old_sse = sse_accum
            sse_accum = 0

            # For each user-movie pair
            for user in range(self.n_users):
                for movie in range(self.n_movies):
                    # if the rating exists
                    if self.user_item_matrix[user, movie] > 0:
                        # compute the error as the actual minus the dot product of the user
                        # and movie latent features
                        prediction = np.dot(user_mat[user], movie_mat[:,movie])
                        diff = self.user_item_matrix[user,movie] - prediction

                        # Keep track of the sum of squared errors for the matrix
                        sse_accum += diff ** 2

                        # update the values in each matrix in the direction of the gradient
                        user_mat[user] += learning_rate * 2 * diff * movie_mat[:, movie]
                        movie_mat[:, movie] += learning_rate * 2 * diff * user_mat[user]

            #print results for iteration
            print("%d \t\t %f" % (i + 1, sse_accum / self.n_ratings))

        # FunkSVD solution
        # Storing the user mat and movie mat
        self.user_mat = user_mat
        self.movie_mat = movie_mat

        # Knowledge base solution
        self.ranked_movies = rf.create_ranked_df(self.movies, self.reviews)

    def predict_rating(self, user_id, movie_id ):
        '''
        makes predictions of a rating for a user on a movie-user combo

        :param user_id: user id (int)
        :param movie_id: movie_id (int)

        :return prediction: The predicted rating for the user-movie (float)
        '''

        # Getting user row and movie column
        user_row = np.where(self.user_ids == user_id)[0][0]
        movie_col = np.where(self.movie_ids == movie_id)[0][0]

        # Dot product of that row and column in U and V to make prediction
        prediction = np.dot(self.user_mat[user_row], self.movie_mat[:,movie_col])

        return prediction

    def make_recs(self,_id, _id_type='movie', rec_num=5):
        '''
        given a user id or a movie that an individual likes
        make recommendations

        :param _id: id of the user or movie (int)
        :param _id_type: "movie" or "user" (str)
        :param rec_num: number of desired recommendations (int)

        :return rec_ids: list of recommended movie ids for @_id_type
        :return rec_names: list of recommended movie names for @_id_type

        '''

        if _id_type == 'user':

            if _id in self.user_item_df.index:
                # Get the index of which row the user is in for use in U matrix
                user_row = np.where(self.user_ids == _id)[0][0]
                # take the dot product of that U row and the V matrix
                predictions = np.dot(self.user_mat[user_row] ,self.movie_mat)
                pred_df = pd.DataFrame(predictions, index=self.movie_ids, columns=['Predictions'])
                # Sorting the pred df to have the best rated movies first
                pred_df.sort_values(by='Predictions', ascending=False, inplace=True)
                rec_ids = pred_df.index[:rec_num]

            else:
                rec_ids = self.ranked_movies['movie_id'][:rec_num]

        if _id_type == 'movie':

            similar_movies = rf.find_similar_movies(_id,self.movies)
            best_movies = pd.DataFrame(self.ranked_movies['movie'].values)
            recs = best_movies[best_movies['movie'].isin(similar_movies)][:rec_num]
            rec_ids = np.array(recs.index)

        rec_names = rf.get_movie_names(rec_ids, self.movies)

        return rec_ids, rec_names

if __name__ == '__main__':
    # test different parts to make sure it works
    recomm = Recommender()

    recomm.fit('Data/train_data.csv', 'Data/movies_clean.csv')

    print("Learning rate, latent features, number of iterations")
    print(recomm.learning_rate, recomm.latent_features, recomm.iter)
    counter = 0
    for user in recomm.user_item_df.index:

        if counter < 15:
            rec_ids, rec_names = recomm.make_recs(user,'user')
            print("For user {}, our recommendations are: \n {}".format(user, rec_names))
            counter += 1
        else:
            break

