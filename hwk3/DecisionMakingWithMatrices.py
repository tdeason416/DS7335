# Decision making with Matrices

import numpy as np
from collections import OrderedDict
import json
 
# This is a pretty simple assingment.  You will do something you do everyday, but today it will be with matrix manipulations.
 
# The problem is: you and your work firends are trying to decide where to go for lunch. You have to pick a resturant thats best for everyone.  Then you should decided if you should split into two groups so eveyone is happier.
 
# Displicte the simplictiy of the process you will need to make decisions regarding how to process the data.
 
# This process was thoughly investigated in the operation research community.  This approah can prove helpful on any number of decsion making problems that are currently not leveraging machine learning.
 
# You asked your 10 work friends to answer a survey. They gave you back the following dictionary object.


class DecisionMakingWithMatrices(object):
    '''
    Enables Decision making with matracies.
    --------
    ATTRIBUTES
    self.people: dict of dicts
        -   human readable representation of people and their preferences
    self.people_rows: list of str
        -   names of participants
    self.people_cols: list of str
        -   names of restaurant descriptions
    self.people_mtrx: np.matrix
        -   linear representation of people and their preferences
    self.restaurants dict of dicts
        -   human readable representation of restaurants and their descriptions
    self.restaurant_rows: list of str
        -   names of restaurants involved
    self.restaurant_cols: list of str
        -   names of restaurant descriptors used
    self.restaurants_matrix: np.matrix
        -   linear representation of restaurants and their descriptions
    --------
    METHODS
    __init__
    describe_lc
    find_restaurant_preferences
    calculate_all_user_prefs
    output_restaurant_scores
    non_parametric_restaurant_scores
    answer_why_diffirent
    find_problematic_users
    misery_score_non_param
    misery_score_numeric
    misery_score_answer
    boss_is_paying
    team_bravo
    '''
    def __init__(self):
        self.people = {
            'Jane': {
                'willingness to travel': 3,
                'desire for new experience': 4,
                'cost': 3,
                'sensitivity to ratings': 3,
                'vegetarian': 1,
                'small plates': 1,
                'italian': 1,
                'seafood': 4,
                'asian' : 5
            },
            'Alan': {
                'willingness to travel': 1,
                'desire for new experience': 2,
                'cost': 4,
                'sensitivity to ratings': 5,
                'vegetarian': 1,
                'small plates': 1,
                'italian': 5,
                'seafood': 2,
                'asian' : 2
            },
            'Kevin': {
                'willingness to travel': 4,
                'desire for new experience': 3,
                'cost': 4,
                'sensitivity to ratings': 3,
                'vegetarian': 2,
                'small plates': 3,
                'italian': 5,
                'seafood': 1,
                'asian' : 1
            },
            'Jeff': {
                'willingness to travel': 5,
                'desire for new experience': 4,
                'sensitivity to ratings': 1,
                'cost': 3,
                'vegetarian': 3,
                'small plates': 5,
                'italian': 2,
                'seafood': 4,
                'asian' : 3
            },
            'Angela':{
                'willingness to travel': 3,
                'desire for new experience': 4,
                'cost': 1,
                'sensitivity to ratings': 2,
                'vegetarian': 3,
                'small plates': 2,
                'italian': 3,
                'seafood': 4,
                'asian' : 5
            },
            'Sam': {
                'willingness to travel': 5,
                'desire for new experience': 5,
                'cost': 3,
                'sensitivity to ratings': 4,
                'vegetarian': 1,
                'small plates': 3,
                'italian': 4,
                'seafood': 2,
                'asian' : 4
            },
            'Dan': {
                'willingness to travel': 4,
                'desire for new experience': 5,
                'cost': 2,
                'sensitivity to ratings': 3,
                'vegetarian': 4,
                'small plates': 4,
                'italian': 1,
                'seafood': 3,
                'asian' : 5
            },
            'Karen': {
                'willingness to travel': 1,
                'desire for new experience': 2,
                'cost': 2,
                'sensitivity to ratings': 5,
                'vegetarian': 3,
                'small plates': 5,
                'italian': 4,
                'seafood': 4,
                'asian' : 2
            },
            'Steph':{
                'willingness to travel': 3,
                'desire for new experience': 4,
                'cost': 1,
                'sensitivity to ratings': 4,
                'vegetarian': 3,
                'small plates': 2,
                'italian': 1,
                'seafood': 4,
                'asian' : 5
            },
            'Doug':{
                'willingness to travel': 1,
                'desire for new experience': 3,
                'cost': 5,
                'sensitivity to ratings': 1,
                'vegetarian': 1,
                'small plates': 3,
                'italian': 5,
                'seafood': 4,
                'asian' : 2
            }
        }
        # Transform the user data into a matrix(M_people). Keep track of column and row ids.
        lst_to_mtrx = []
        self.people_rows = []
        self.people_cols = [
            'willingness to travel',
            'desire for new experience',
            'cost',
            'sensitivity to ratings',
            'vegetarian',
            'small plates',
            'italian',
            'seafood',
            'asian'
        ]
        for person, description in self.people.items():
            self.people_rows.append(person)
            lst_to_mtrx.append([description[cat] for cat in self.people_cols])
        self.people_mtrx =  np.matrix(lst_to_mtrx)
        # Next you collected data from an internet website. You got the following information.
        ### Note the catagories for 'cost' and 'distance' are large numbers for low costs and small distances 
        ### respectivly. This is done for consistancy for matrix multiplcation.  All values are also from 1-5
        self.resturants  = {
            'Purple Cafe': {
                'distance' : 3.5,
                'novelty' : 3,
                'cost': 2,
                'average rating': 4.5,
                'cusine': 'small plates'
            },
            "Matt's At the Market": {
                'distance': 4,
                'novelty': 3,
                'cost': 1,
                'average rating': 4.5,
                'cusine': 'seafood'
            },
            "Elliott's Oyster House": {
                'distance' : 3,
                'novelty': 2,
                'cost': 2,
                'average rating': 4,
                'cusine': 'seafood'
            },
            "Ivar's Acres of Clams": {
                'distance' : 2,
                'novelty': 1,
                'cost': 4,
                'average rating': 3,
                'cusine': 'seafood'
            },
            "Pink Door": {
                'distance' : 4,
                'novelty': 5,
                'cost': 2,
                'average rating': 4,
                'cusine': 'italian'
            },
            "Veggie Grill": {
                'distance' : 3.5,
                'novelty': 1,
                'cost': 4,
                'average rating': 3,
                'cusine': 'vegetarian'
            },
            "Outlier": {
                'distance' : 3.5,
                'novelty': 5,
                'cost': 1,
                'average rating': 2,
                'cusine': 'small plates'
            },
            "Umma's Lunch Box": {
                'distance' : 2.5,
                'novelty': 2,
                'cost': 4,
                'average rating': 4.5,
                'cusine': 'asian'
            },
            "Il Corvo": {
                'distance' : 1,
                'novelty': 4,
                'cost': 4,
                'average rating': 5,
                'cusine': 'italian'
            },
            "Okinawa": {
                'distance' : 2.5,
                'novelty': 1,
                'cost': 5,
                'average rating': 4,
                'cusine': 'asian'
            },
            "Evergreens": {
                'distance' : 3,
                'novelty': 1,
                'cost': 4,
                'average rating': 4,
                'cusine': 'vegetarian'
            }
        }
        # Transform the restaurant data into a matrix(M_resturants) use the same column index.
        lst_to_mtrx_r = []
        self.resturant_rows = []
        self.resturant_cols = [
            'distance',
            'novelty',
            'cost',
            'average rating',
            'cusine: vegetarian',
            'cusine: small plates',
            'cusine: italian',
            'cusine: seafood',
            'cusine: asian'
        ]
        for restaurant, description in self.resturants.items():
            self.resturant_rows.append(restaurant)
            row = [description[cat] for cat in self.resturant_cols[:4]]
            for cat in self.resturant_cols[4:]:
                if cat.split(':')[-1].strip() != description['cusine']:
                    row.append(5)
                else:
                    row.append(0)
            lst_to_mtrx_r.append(row)
        self.restaurant_mtrx =  np.matrix(lst_to_mtrx_r)

    def describe_lc(self):
        '''
        The most imporant idea in this project is the idea of a linear combination.
        Informally describe what a linear combination is and how it will relate to our resturant matrix.
        '''
        lc_answer = '''A linear combination is the act of combining two or more components of N dimensionality to calculate how they would interact.  One of the most simple applications of a lienar combination is the act of solving a system of linear equations to minimize the value of a system in machine learning or system dynamics.  Linear combinations require inputs and outputs to be matched.  In the case of our lunch preference equations, this means the values input for restaurant description must match the restaurant preferences of the survey participants.'''
        print(lc_answer)

    def find_restaurant_preferences(self, name):
        '''
        Choose a person and compute(using a linear combination) the top restaurant for them.  What does each entry in the resulting vector represent.
        --------
        ANSWER
            This method gives a users preferences for each of the restaurants in the survey
        '''
        person = self.people[name]
        p = np.matrix([person[col] for col in self.people_cols])
        prefs = np.dot(p, self.restaurant_mtrx.T).tolist()[0]
        return {k:v for k,v in zip(self.resturant_rows, prefs)}

    def calculate_all_user_prefs(self):
        '''
        Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent?
        --------
        ANSWER
            The output of this matrix represents the user score for each user on each restaurant
        '''
        return np.dot(self.people_mtrx, self.restaurant_mtrx.T)
    
    def _vals_to_ranks(self, lst):
        '''
        Takes a list and outputs their reverse ranks (where larger numbers are prefereable)
        --------
        PARAMETERS
        lst: array like (numeric)
        --------
        RETURNS
        n_vals: list (integers)
            -   Non-parametric representation of the original 'lst'
        '''
        n_vals = []
        preferences = {v:r for r,v in enumerate(sorted(lst))}
        for val in lst:
            n_vals.append(preferences[val])
        return n_vals

    def output_restaurant_scores(self):
        '''
        Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?
        --------
        ANSWER
            The sum of the columns in the M_usr_x_rest matrix represent the relative score of each restaurant for the whole group (the restaurant with the highest score will be the most prefered to the group)
        '''
        M_usr_x_rest = self.calculate_all_user_prefs()
        output_mtrx = {}
        scores = [M_usr_x_rest[:,n].sum() for n in range(M_usr_x_rest.shape[1])]
        ranks = self._vals_to_ranks(scores)
        return { rsnt: score for rsnt, score in zip(self.resturant_rows, ranks) }

    def non_parametric_restaurant_scores(self):
        '''
        Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   Do the same as above to generate the optimal resturant choice.
        '''
        nonpar_output = {}
        M_usr_x_rest = self.calculate_all_user_prefs()
        M_usr_x_rest_rank = \
            np.matrix([self._vals_to_ranks(M_usr_x_rest[row].tolist()[0])
                                for row in range(M_usr_x_rest.shape[0])])
        scores = [M_usr_x_rest_rank[:,n].sum() for n in range(M_usr_x_rest_rank.shape[1])]
        ranks = self._vals_to_ranks(scores)
        return { rsnt: score for rsnt, score in zip(self.resturant_rows, ranks) }
    
    def answer_why_diffirent(self):
        '''
        Why is there a difference between the two?  What problem arrives?  What does represent in the real world?
        
        How should you preprocess your data to remove this problem.
        '''
        answer = '''
            Summing all of the scores using method 1 allows one person's preferences to dominate the overall score because that one person may strongly favor a particular resturant.  By making the scores non-parametric (adding the ranks of each resturant instead of their raw scores) by method 2, Each participant's rank vote is counted equally.  The result in my survey is that the same places are chosen for first and last by the group, but the other ranks are jumbled by using this diffirent method.  Both methods may have their value in the real world, using method 2 is more likley to result in the group going to a place that most people like; while a single person could feel very strongly against it.  Method 1 is more likley to result in the group going to a place which most participants do not have a strong opponion of; while a single person may have a very strong preference toward that place.  In other words, using a non-parametric system makes it harder for a single person to manipulate the system toward a specific preference.\n
            It could be possible to reduce the impact of agressive users by applying weights to the preferences matrix.
            '''
        print(answer)

    def find_problematic_users(self):
        '''
        Find user profiles that are problematic, explain why?
        --------
        ANSWER
            By checking to see what the average distance from a score of 3 each user inputs, we find that a couple users have an average of 1.556 and 1.333; meaning these users mostly vote in 1's and 5's; while other users have numbers around 1.  The lowest of which is a score of .889.  The user who's average distance from 3 is 1.556 will have almost twice the impact on the outcome as the user who's score here is .889
        '''
        user_deltas = {}
        for person, preferences in self.people.items():
            scores = np.array(list(preferences.values()))
            dist_from_3 = np.abs(scores - 3)
            user_deltas[person] = dist_from_3.mean()
        return user_deltas
 
    def misery_score_non_param(self):
        '''
        First method of computing misery score this method counts the users who will end up going to a restaurant which is on the bottom three of their choices as being miserable
        '''
        M_usr_x_rest = self.calculate_all_user_prefs()
        M_usr_x_rest_rank = \
            np.matrix([self._vals_to_ranks(M_usr_x_rest[row].tolist()[0])
                                for row in range(M_usr_x_rest.shape[0])])
        less_then_4 = M_usr_x_rest_rank < 4
        scores = [less_then_4[:,n].sum() for n in range(M_usr_x_rest_rank.shape[1])]
        return { rsnt: score for rsnt, score in zip(self.resturant_rows, scores) }

    def misery_score_numeric(self):
        '''
        Second method of computing misery score
        '''
        M_usr_x_rest = self.calculate_all_user_prefs()
        meh_score = np.dot(
            np.ones(self.people_mtrx.size).reshape(*self.people_mtrx.shape) * 3,
            self.restaurant_mtrx.T
        )
        shifted = M_usr_x_rest - meh_score
        shifted[shifted > 0] = 0
        scores = [shifted[:,n].sum() for n in range(shifted.shape[1])]
        return { rsnt: score for rsnt, score in zip(self.resturant_rows, scores) }

    def misery_score_answer(self):
        '''
        Think of two metrics to compute the disatistifaction with the group.

        Should you split in two groups today?
        --------
        ANSWER

        '''
        answer = '''
            These two methods give very diffirent values, but in both cases, the most highly ranked resturant of both methods is gievn a high misery index.  Using the numeric misery index, the most highly ranked restaurant (pink door), gets the highest misery index.  Since this method takes into effect how much a person disslikes a place, those that do not want to go to Pink Door, seem to very much be against the idea.  This lends some credibility to the arguement that lunch should be split into multipule groups.
        '''
        print(answer)

    def boss_is_paying(self):
        '''
        Ok. Now you just found out the boss is paying for the meal. How should you adjust. Now what is best restaurant?
        --------
        The first restaurant remains pink door, but the 2nd through 4th restaurants change dramatically in rank; so there would be an impact if we took the misery index of the pink door into account for the group.
        '''
        rests = self.restaurant_mtrx[:,[0,1,3,4,5,6,7,8]]
        ppl = self.people_mtrx[:,[0,2,3,4,5,6,7,8]]
        M_usr_x_rest_nocost = np.dot(ppl, rests.T)
        output_mtrx = {}
        scores = [M_usr_x_rest_nocost[:,n].sum() for n in range(M_usr_x_rest_nocost.shape[1])]
        ranks = self._vals_to_ranks(scores)
        return { rsnt: score for rsnt, score in zip(self.resturant_rows, ranks) }
 

    def team_bravo(self):
        '''
        Tommorow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  Can you find their weight matrix?
        '''
        answer = '''
            No, I cant find their weight matrix just based on their preferences.  This is because unrecoverable information is lost when converting the output from the restaurant scores matrix to group ranks, but i could find the weight matrix if I was given each participant's score for each resturaunt.  This is because matrix multiplication is reversable.
        '''
        print(answer)

if __name__ == "__main__":
    dm = DecisionMakingWithMatrices()
    dm.describe_lc()
    print("Doug's scores are:")
    for rest, score in dm.find_restaurant_preferences('Doug').items():
        print('\t', rest, ": ", score)
    dm.calculate_all_user_prefs()
    print("The group scores are:")
    for rest, score in dm.output_restaurant_scores().items():
        print('\t', rest, ": ", score)
    print("The composite group scores are:")
    for rest, score in dm.non_parametric_restaurant_scores().items():
        print('\t', rest, ": ", score)
    dm.answer_why_diffirent()
    print('Each participant has the following leverage:')
    for person, score in dm.find_problematic_users().items():
        print("\t", person, ':', score)
    print('Using the rank method, the misery score for each restaurant is:')
    for rest, score in dm.misery_score_non_param().items():
        print("\t", rest, ':', score)
    print('Using the points method, the misery score for each restaurant is:')
    for rest, score in dm.misery_score_numeric().items():
        print("\t", rest, ':', score)
    dm.misery_score_answer()
    print('Checking on restaurant ranks when the boss is paying:')
    for rest, score in dm.boss_is_paying().items():
        print("\t", rest, ':', score)
    dm.team_bravo()
    