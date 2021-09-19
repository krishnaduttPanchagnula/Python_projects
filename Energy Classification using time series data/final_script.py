# import the important libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler


# import the csv file

class voltaware():
    def __init__(self) -> None:
        print('You are being initalized..')
      
    def load_data(self):
        csv_filepath = input("Please enter a valid file path to a csv: ")

        pd.options.display.max_columns = None

        while not os.path.isfile(csv_filepath):
            print("Error: That is not a valid file, try again...")
            csv_filepath = input("Please enter a valid file path to a csv: ")

        try:
            df = pd.read_csv(csv_filepath)
            
            # Add your other code to manipulate the dataframe read from the csv here
        except BaseException as exception:
            print(f"An exception occurred: {exception}")

        # sorting data according to timestamp
        df['time'] = pd.to_datetime(df['timestamp'])
        df.sort_values(by = ['timestamp'])

        # Filling missing values with interpolate

        self.cleaned = df.interpolate(method='from_derivatives', limit = 30)

        self.merged = pd.merge_ordered(df,self.cleaned, on='event_id',suffixes=('_original', '_transformed'))

        # Viewing the old and new dataframes togehter
        print(self.merged)
    def process(self):

        # Calculating transient states for each of the peak states
        colnames = ['event_id','peak_1', 'peak_2', 'peak_3', 'peak_4', 'peak_5', 'peak_6', 'peak_7','peak_8', 'peak_9','time','current'] 
        self.process = self.cleaned[colnames]

        for ind, row in self.process.iterrows():
            for i in range(1,10):
                self.process.loc[ind, 'transient' + '_{}'.format(i)] = row ['peak_{}'.format(i)] - row['current']

        #mean  and std of transients
        tr = ['transient_1', 'transient_2',
            'transient_3', 'transient_4', 'transient_5', 'transient_6',
            'transient_7', 'transient_8', 'transient_9']
        self.process['mean_transient'] = np.mean(self.process[tr],axis = 1)
        self.process['std_transient'] = np.std(self.process[tr],axis = 1)

        #FInding the mean of top 7 transients

        self.pre_sort = self.process[tr].to_numpy() #converting to numpy
        order = np.argsort(self.pre_sort,axis =1) # finding the ordered indices

        final = []
        mean_arr = [] # creating empty lists to hold the data


        for  i in range(len(order)):
            sorted_array = self.pre_sort[i][order[i]]
            final += [sorted_array]
            mean = np.mean(final[i][-7:])
            mean_arr += [mean]

        self.process['mean_top 7 transients'] = mean_arr

        print(self.process)
    
    def normalise(self):
        
        #Normalization of columns 

        for ind, row in self.process.iterrows():
            for i in range(1,10):
                self.process.loc[ind, 'transient_sq' + '_{}'.format(i)] = row ['transient_{}'.format(i)] ** 2

        rt = ['transient_sq_1',
            'transient_sq_2', 'transient_sq_3', 'transient_sq_4', 'transient_sq_5',
            'transient_sq_6', 'transient_sq_7', 'transient_sq_8', 'transient_sq_9']

        rtr = self.process[rt].sum(axis=1)
        
        self.process['sqrt'] =  rtr ** 0.5

        #handling zero errors and Normalzing the values

        for ind, row in self.process.iterrows():
            for i in range(1,10):
                try:
                    self.process.loc[ind, 'normalised_transient' + '_{}'.format(i)] = row ['transient_{}'.format(i)] /row['sqrt']
                except ZeroDivisionError:
                    self.process.loc[ind, 'normalised_transient' + '_{}'.format(i)] = 0


        print(self.process)
        self.comb = pd.merge(self.process,self.cleaned,on =['event_id'],how ='outer')

    def show_features(self):
        
        # Shows the graph for the list

        vis_col = []
        # Input total number of features to be plotted
        n = int(input("Enter total number of features to be plotted : "))
        # Enter elements separated by comma
        vis_col = list(map(str,input("Enter the column names from {} : ".format (self.comb.columns)).strip().split(',')))[:n]
        print("The entered list is: \n",vis_col)

        plt.figure (figsize=(20,20))
        plt.title('{} vs time graph'.format(vis_col))
        plt.xlabel('Time')
        plt.ylabel(vis_col)
        sns.set_theme()
        cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

        for x in vis_col:
            sns.scatterplot(x=self.comb['time_x'], y= self.comb[x],size=self.comb[x],
            sizes=(40, 400), alpha=.5, palette=cmap)
            
            plt.show()

    def viz(self):
        teap = ['active_power','reactive_power','current_y']
        mi = MinMaxScaler()

        for x in teap:
            self.comb[x + '_{}'.format('norm')] =  mi.fit_transform(self.comb[[x]])
        graph = ['active_power_norm', 'reactive_power_norm', 'current_y_norm']


        cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

        plt.figure (figsize=(30,30))
        
        for item in graph:
            sns.lineplot(x = self.comb['time_x'], y = self.comb[item],alpha=.5, palette=cmap)
            
        plt.show()






