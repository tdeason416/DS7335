#!/usr/bin/python
import numpy as np
import json
import re
import os
import matplotlib.pyplot as plt
from collections import Counter
### DeathToGridSearch is from hwk1 and must be installed or added to $PYTHONPATH
from DeathtoGridSearch import GridSearch



### Running Hwk2.py will answer all questions and generate an output of plots with .csv files for all tested models.
### This will take a couple hours (due to the Grid Search Object)
####################################################################################################################

class Homework2(object):
    '''
    Class for completing the tasks in homework2
    --------
    Attributes
    self.columns: list of strings
    self.data: list of lists (numeric)
    self.colmap: dict { str : int }
    self.colmap_r dict { int : str }x
    self.j_idxs: list of ints (numeric)
        -   indexes for subset of data where procedure code is J
    self.jdf: list of lists (numeric)
        -   subset of data where procedure code is J
    --------
    Methods
    answer_question_1a
        -   Find the number of claim lines that have J-codes.
    answer_question_1b
        -   How much was paid for J-codes to providers for 'in network' claims?
    answer_question_1c
        -   What are the top five J-codes based on the payment to providers?
    answer_question_2a
        -   Create a scatter plot that displays the number of unpaid claims (lines where the
             ‘Provider.Payment.Amount’ field is equal to zero) for each provider versus the number of paid claims.
    answer_question_2b
        -   What insights can you suggest from the graph?
    answer_question_2c
        -   Based on the graph, is the behavior of any of the providers concerning? Explain.
    answer_question_3a
        -   What percentage of J-code claim lines were unpaid?
    answer_question_3b
        -   Create a model to predict when a J-code is unpaid. Explain why you choose the   
            modeling approach.
    answer_question_3c
        -   How accurate is your model at predicting unpaid claims?
    answer_question_3d
        -   What data attributes are predominately influencing the rate of non-payment?
    answer_all_questions
        -   answer all of the above questions in sequence
    '''
    def __init__(self, filename="claim.sample.csv"):
        '''
        Load data into python and make readable
        '''
        raw_data = [re.sub("[^a-zA-Z0-9.,]", "", i).split(',') 
                    for i in open(filename).read().strip().split("\n")]
        ## Create seperate objects for data and columns
        self.columns = raw_data[0]
        self.data = raw_data[1:]
        #map columns to the number which they are in
        self.colmap = {k:v for v,k in enumerate(self.columns)}
        self.colmap_r = {v:k for k,v in self.colmap.items()}
        self.j_idxs = []
        self.jdf = []

    ### QUESTION 1
    ###############

    def answer_question_1a(self):
        '''
        Find the number of claim lines that have J-codes.
        --------
        ANSWER
            The number of J-Codes is:  51029
        '''
        j_codes = 0

        for rownum, row in enumerate(self.data):
            try:
                if row[self.colmap['Procedure.Code']][0] == 'J':
                    j_codes += 1
                    self.j_idxs.append(rownum)
                    self.jdf.append(row)
            except IndexError:
                continue
        print("QUESTION 1: A medical claim is denoted by a claim number ('Claim.Number'). Each claim consists of one or more medical lines denoted by a claim line number ('Claim.Line.Number').")
        print("\t a.) Find the number of claim lines that have J-codes.")
        print("\t\t The number of J-Codes is: ", j_codes)

    def answer_question_1b(self):
        '''
        How much was paid for J-codes to providers for 'in network' claims?
        --------
        ANSWER
            $ 2418429.57 Was paid for "in network" claims
        '''
        total_payment = 0
        for rownum in self.j_idxs:
            total_payment += float(self.data[rownum][self.colmap["Provider.Payment.Amount"]])
        print(total_payment)
        print("\t b.) How much was paid for J-codes to providers for 'in network' claims?")
        print("\t\t $ {:.2f} was paid for 'in network' claims".format(total_payment))

    def answer_question_1c(self):
        '''
        What are the top five J-codes based on the payment to providers?
        --------
        ANSWER
            J1644  :  81909.39601500003
            J3490  :  90249.91244999997
            J9310  :  168630.87357999996
            J0180  :  299776.56076499994
            J1745  :  434232.08058999997
        '''
        codes_map = {}
        for rownum in self.j_idxs:
            try:
                codes_map[self.data[rownum][self.colmap["Procedure.Code"]]] += \
                        float(self.data[rownum][self.colmap["Provider.Payment.Amount"]])
            except KeyError:
                codes_map[self.data[rownum][self.colmap["Procedure.Code"]]] = \
                        float(self.data[rownum][self.colmap["Provider.Payment.Amount"]])
        codes_inv = {v:k for k,v in codes_map.items()}

        print("\t c.) What are the top five J-codes based on the payment to providers?")
        for val in sorted(codes_map.values())[-5:]:
            print("\t\t", codes_inv[val], " : ", val)

    ### QUESTION 2
    ###############

    def answer_question_2a(self):
        '''
        Create a scatter plot that displays the number of unpaid claims (lines where the
        ‘Provider.Payment.Amount’ field is equal to zero) for each provider versus the number of paid claims.
        --------
        ANSWER
            Plot is located at q2a.png
        '''
        unpaid_claims = Counter()
        paid_claims = Counter()

        for row in self.jdf:
            if int(float(row[self.colmap["Provider.Payment.Amount"]])) == 0:
                unpaid_claims[row[self.colmap["Provider.ID"]]] += 1
            else:
                paid_claims[row[self.colmap["Provider.ID"]]] += 1

        x = [paid_claims[k] for k in unpaid_claims.keys()]
        y = [unpaid_claims[k] for k in unpaid_claims.keys()]

        fig, ax = plt.subplots(1,1, figsize=(15,15))
        ax.scatter(x, y)
        ax.set_title("Paid Claims vs Unpaid Claims")
        ax.set_xlabel("Paid Claims")
        ax.set_ylabel("Unpaid Claims")
        ax.set_ylim(-5,15000)
        ax.set_xlim(-5,15000)

        try:
            fig.savefig('q2a.png')
        except IOError:
            os.mkdir('charts')
            fig.savefig('q2a.png')

        print('''QUESTION 2. For the following exercises, determine the number of providers that were paid for at least one J-code. Use the J-code claims for these providers to complete the following exercises.''')
        print('''\t a.) Create a scatter plot that displays the number of unpaid claims (lines where the ‘Provider.Payment.Amount’ field is equal to zero) for each provider versus the number of paid claims.''')
        print("\t\t plot is avaliable at 'charts/q2a.png'")

    def answer_question_2b(self):
        '''
        What insights can you suggest from the graph?
        --------
        ANSWER
            There apears to be a somewhat linear relationship between the number of Paid claims and the number of unpaid claims, but in almost all cases, unpaid claims outnumber paid claims significently.
        '''
        print("\t b.) What insights can you suggest from the graph?")
        print("\t\t There apears to be a somewhat linear relationship between the number of Paid claims and the number of unpaid claims, but in almost all cases, unpaid claims outnumber paid claims significently.")

    def answer_question_2c(self):
        '''
        Based on the graph, is the behavior of any of the providers concerning? Explain.
        --------
        ANSWER
            Based on the graph, it appears that most providers have significently more unpaid claims then paid, claims, this would seem to be an indication that this buisness is not equitable and the provider will soon go out of buisness.
        '''
        print("\t c.) Based on the graph, is the behavior of any of the providers concerning? Explain.")
        print("\t\tBased on the graph, it appears that most providers have significently more unpaid claims then paid, claims, this would seem to be an indication that this buisness is not equitable and the provider will soon go out of buisness.")
    
    ### QUESTION 3
    ###############

    def answer_question_3a(self):
        '''
        What percentage of J-code claim lines were unpaid?
        --------
        ANSWER
            88.30% of all claims were unpaid
        '''
        unpaid_count = 0
        paid_count = 0

        for row in self.jdf:
            if int(float(row[self.colmap["Provider.Payment.Amount"]])) == 0:
                unpaid_count += 1
            else:
                paid_count += 1
        unpaid_ratio = unpaid_count / (unpaid_count + paid_count) * 100
        print('''QUESTION 3 Consider all claim lines with a J-code.''')
        print('''\t a.) What percentage of J-code claim lines were unpaid?''')
        print("\t\t {:.2f}% of all claims were unpaid".format(unpaid_ratio))

    def answer_question_3b(self):
        '''
        Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.
        --------
        ANSWER
            I used the grid search method I created in hwk1
        '''
        ## Seperate into Numeric columns and Catagorical columns (which will need dummies)
        model_cols = []
        numeric_cols = [
            'Subscriber.Payment.Amount',
            'Claim.Charge.Amount',
        ]
        dummy_cols = {
            "Provider.ID" : {},
            "Line.Of.Business.ID" : {},
            "Service.Code": {},
            "In.Out.Of.Network" : {},
            "Network.ID": {},
            "Agreement.ID" : {},
            "Price.Index": {},
            "Claim.Type": {},
            "Procedure.Code": {},
            "Revenue.Code": {}
        }
        ## Generate Dummy Cols
        for rownum, row in enumerate(self.jdf):
            for col in dummy_cols.keys():
                try:
                    dummy_cols[col][rownum].add(row[self.colmap[col]])
                except KeyError:
                    dummy_cols[col][row[self.colmap[col]]] = {rownum}
        for colname, dumdict in dummy_cols.items():
            for dummy in dumdict.keys():
                model_cols.append("{}${}".format(colname, dummy))
        
        ## Add dummy cols to new data set (model_df)
        model_df = []
        for idx, row in enumerate(self.jdf):
            ith_row = []
            for colname in model_cols:
                col, val = colname.split("$")
                if idx in dummy_cols[col][val]:
                    ith_row.append(1)
                else:
                    ith_row.append(0)
            for numeric in numeric_cols:
                ith_row.append(float(row[self.colmap[numeric]]))
            ## Convert Provider Payment amount to binary classified value
            ## And remove this column to ensure no y-leakage exists within the model
            if float(row[self.colmap["Provider.Payment.Amount"]]) > 0.00:
                ith_row.append(0.0)
            else:
                ith_row.append(1.0)
            model_df.append(ith_row)
        
        ## Retain column names for future reference
        for numeric_col in numeric_cols:
            model_cols.append(numeric_col)
        
        ## Turn model_df into numpy array to use when modeling
        np_df = np.array(model_df)

        ## Identify where label cols are true and false and create balanced class sizes
        ## balanced here being the number of paid claims == number of unpaid claims
        true_vals = np_df[np_df[:,-1] == 1.0]
        false_vals = np_df[np_df[:,-1] == 0.0]
        b_df = np.concatenate([false_vals, true_vals[:false_vals.shape[0]]])

        ## Dataset is now ready for the function generated in DeathtoGridSearch
        gs = GridSearch(b_df[:,:-1], b_df[:,-1])
        gs.optimize_all_models()
        gs.save_results()

        print('''\t b.) Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.''')
        
        print('''\t\t I noticed some of the columns weere catagorical, so I removed these columns and generated dummy values for modeling reasons.  I then created balanced clas sizes by sampling the True (or did not pay) values.  This dramatically decreases the size of my dataset.  Using the methods I previously developed on the DeathToGridSearch exersize, I cross validated all of the models using 5 folds and checked many tunable parameters.''')

    def answer_question_3c(self):
        '''
        How accurate is your model at predicting unpaid claims?
        --------
        ANSWER
            The model is pretty accurate at predicting unpaid claims.  Overall the best model was a XGBoosted classifier with a max depth of 43 and 101 estimators achieves an overall accuracy of 92.18% when tested across a 5 fold cross validated set with balanced classes (the unpaid claims were sampled).  Depending on the use case of this model, this accuracy could be enough to flag some cases for potential missed payments.  Since these classes are highly imbalanced, using a high threshold value for misspayment will likley missclassify more then 8% of the total observations.
        '''
        print('\t c.) How accurate is your model at predicting unpaid claims?')
        print('\t\t The model is pretty accurate at predicting unpaid claims.  Overall the best model was a XGBoosted classifier with a max depth of 43 and 101 estimators achieves an overall accuracy of 92.18% when tested across a 5 fold cross validated set with balanced classes (the unpaid claims were sampled).  Depending on the use case of this model, this accuracy could be enough to flag some cases for potential missed payments.  Since these classes are highly imbalanced, using a high threshold value for misspayment will likley missclassify more then 8 percent of the total observations.')

    def answer_question_3d(self):
        '''
        What data attributes are predominately influencing the rate of non-payment?
        --------
        ANSWER
            Since the model which produced the best performance was a boosed ensemble model, true feature importances can not be interpeted from the model, 
        '''
        return None

    ### ALL QUESTIONS
    #################

    def answer_all_questions(self):
        self.answer_question_1a()
        self.answer_question_1b()
        self.answer_question_1c()
        self.answer_question_2a()
        self.answer_question_2b()
        self.answer_question_2c()
        self.answer_question_3a()
        self.answer_question_3b()
        self.answer_question_3c()
        self.answer_question_3d()

if __name__ == "__main__":
    Hwk2Solver = Homework2()
    Hwk2Solver.answer_all_questions()
        
