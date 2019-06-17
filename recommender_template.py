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



    def fit(self, reviews_loc, movies_loc, latent_features, n_iter, learning_rate  ):
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
        user_items = self.reviews[['user_id', 'movie_id', 'rating', 'timestamp']]
        user_by_movie = user_items.groupby(['user_id', 'movie_id'])['rating'].max().unstack()

        self.user_item_matrix = np.array(user_by_movie)
        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iter = n_iter

        # Set up useful values to be used through the rest of the function
        self.n_users = self.user_item_matrix.shape[0]
        self.n_movies = self.user_item_matrix.shape[1]
        self.n_ratings = np.count_nonzero(~np.isnan(self.user_item_matrix))
        self.movie_ids = np.array(user_by_movie.columns)
        self.user_ids = np.array(user_by_movie.index)

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
                    if np.isnan(self.user_item_matrix[user, movie]) == False:
                        # compute the error as the actual minus the dot product of the user
                        # and movie latent features
                        prediction = np.dot(user_mat[user], movie_mat[:,movie])
                        diff = self.user_item_matrix[user,movie] - prediction

                        # Keep track of the sum of squared errors for the matrix
                        sse_accum += diff ** 2

                        # update the values in each matrix in the direction of the gradient
                        user_mat[user] += learning_rate * 2 * diff * movie_mat[:, movie]
                        movie_mat[:, movie] += learning_rate * 2 * diff * user_mat[user]

        # print results for iteration
        print("%d \t\t %f" % (i + 1, sse_accum / self.n_ratings))

        # FunkSVD solution
        # Storing the user mat and movie mat
        self.user_mat = user_mat
        self.movie_mat = movie_mat

        # Knowledge base solution
        self.ranked_movies = rf.create_ranked_df(self.movies, self.reviews)

    def predict_rating(self, ):
        '''
        makes predictions of a rating for a user on a movie-user combo
        '''

    def make_recs(self,):
        '''
        given a user id or a movie that an individual likes
        make recommendations
        '''


if __name__ == '__main__':
    # test different parts to make sure it works
