import os, glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.spatial
from lcs_utils import TDKS_uniform_circle
pd.options.mode.chained_assignment = None  # default='warn'

class lcs_points:
    '''
    Dependencies:
    os, glob, matplotlib.pyplot (as plt), pandas (as pd), numpy (as np), scipy.spatial, TDKS_uniform_circle
    '''
    def __init__(self,
                 metadata_fpath = os.path.join('..','metadata.csv'),
                 prediction_fpath = os.path.join('..','predictions.csv'),
                ):
        '''
        Makes instance of class handling the organisation of metadata for the replication code of the LCS dataset.
        ======
        Inputs
        ======
        metadata_fpath : str
            Path to the metadata CSV file.
        prediction_fpath : str
            path to the predictions CSV file.
        '''
        self.metadata_fpath = metadata_fpath
        self.metadata = pd.read_csv(self.metadata_fpath)
        self.prediction_fpath = prediction_fpath
        self.predictions = {lcs_id: df.reset_index(drop=True) for lcs_id, df in  pd.read_csv(prediction_fpath).groupby('lcs_id')} # Each lcs_id leads to a dataframe
        
        ## GET PREDICTIONS FOR 1-MIN LONG SEGEMENTS AS 4 COORDINATES
        self.paired_predictions = {} # Saved as two consecutive points 30 seconds apart.
        for prediction_fpath, df in self.predictions.items():
            df_isopl_mean = df['isopl_mean']
            df_isoev_mean = df['isoev_mean']
            
            # Make dataframe
            # Jump 30 indices because each half is 30 seconds and predictions were done with 1s hop.
            self.paired_predictions[prediction_fpath] = pd.DataFrame(
                {
                    'start_idxs': df.start_idxs[ :-30],
                    'isopl_1st_half': df_isopl_mean.iloc[  :-30].reset_index(drop=True),
                    'isoev_1st_half': df_isoev_mean.iloc[  :-30].reset_index(drop=True),
                    'isopl_2nd_half': df_isopl_mean.iloc[30:  ].reset_index(drop=True),
                    'isoev_2nd_half': df_isoev_mean.iloc[30:  ].reset_index(drop=True)
                }
            )
        
        ## GET PREDICTIONS FOR 1-MIN LONG SEGMENTS ONLY FOR POINTS BELONGING IN EXPECTED QUADRANTS
        ## IF NO POINTS BELONG THEN LEAVE ALL POINTS IN TO BE SAFE
        self.paired_predictions_in_quadrants = {}
        for prediction_fpath, df in self.paired_predictions.items():
            lcs_id = prediction_fpath.split(os.sep)[-1].split('.')[0]
            s5_type = self.metadata.query('lcs_id == @lcs_id')['s5_type'].iloc[0]

            if s5_type == 'F&E':
                points_in_quadrant = df[(df['isopl_1st_half'] >= 0) & (df['isoev_1st_half'] >= 0) & (df['isopl_2nd_half'] >= 0) & (df['isoev_2nd_half'] >= 0)]
            elif s5_type == 'C&R':
                points_in_quadrant = df[(df['isopl_1st_half'] <= 0) & (df['isoev_1st_half'] >= 0) & (df['isopl_2nd_half'] <= 0) & (df['isoev_2nd_half'] >= 0)]
            elif s5_type == 'B&L':
                points_in_quadrant = df[(df['isopl_1st_half'] <= 0) & (df['isoev_1st_half'] <= 0) & (df['isopl_2nd_half'] <= 0) & (df['isoev_2nd_half'] <= 0)]
            elif s5_type == 'C&T':
                points_in_quadrant = df[(df['isopl_1st_half'] >= 0) & (df['isoev_1st_half'] <= 0) & (df['isopl_2nd_half'] >= 0) & (df['isoev_2nd_half'] <= 0)]

            if len(points_in_quadrant) == 0: # if no points left after restriction, leave all points as "in quadrant" to broaden possibilities of choice of final set of points
                self.paired_predictions_in_quadrants[prediction_fpath] = df
            else:
                self.paired_predictions_in_quadrants[prediction_fpath] = points_in_quadrant
                 
    def plot_points(self,
                    plot_type = 'all_together',
                    max_abs = 0.5,
                    figsize = (8,8),
                    fontsize = 16,
                    verbose = 0):
        '''
        Makes visualisations of the ISO Pleasantness/Eventfulness predictions in the LCS dataset
        
        ======
        Inputs
        ======
        plot_type : str in ['all_together','all_separate','first'] + self.predictions.keys()
            If 'all_together', plots all predictions in self.predictions.keys() in the same plot.
            If 'all_separate', plots all predictions in self.predictions.keys() in different plots.
            If 'first', plots all predictions in self.predictions.keys()[0] only.
            If a value in self.predictions.keys(), plots all predictions for only that file.
        max_abs : float
            Maximum absolute value attained by x- and y- axes in plot(s).
        figsize : tuple of float
            The figure dimensions.
        fontsize : float
            The font size of the figure
        verbose : bool
            Whether to print additional information.            
        '''
        if plot_type not in ['all_together','all_separate','first'] + list(self.predictions.keys()):
            print(f'Warning: Unexpected plot_type. Got plot_type = {plot_type}.')
        
        plt.figure(figsize = figsize)
        
        for idx, prediction_fpath in enumerate(self.predictions.keys()):
            if plot_type == 'first' and idx > 0:
                continue
            elif (plot_type not in ['first','all_together','all_separate']) and (plot_type != prediction_fpath):
                continue
                
            lcs_id = prediction_fpath.split(os.sep)[-1].split('.')[0]
            s5_type = self.metadata.query('lcs_id == @lcs_id')['s5_type'].iloc[0]
            lcs_description = self.metadata.query('lcs_id == @lcs_id')['lcs_description'].iloc[0]

            df = self.predictions[prediction_fpath]
            df_isopl_mean = df['isopl_mean']
            df_isoev_mean = df['isoev_mean']

            points = np.stack((df_isopl_mean.to_numpy(), df_isoev_mean.to_numpy())).T
            hull = scipy.spatial.ConvexHull(points)
            
            if verbose:
                print(f'Mean of points: {np.mean(points, axis = 0):.4f}, median of points: {np.median(points, axis = 0):.4f}')
                print(f'Volume of convex hull: {hull.volume:.4f}, TDKS test statistic: {TDKS_uniform_circle(points[:,0],points[:,1]):.4f}')

            if plot_type == 'all_separate':
                plt.figure(figsize = figsize)
            
            if plot_type == 'all_together':
                plt.title(f'All soundscapes', fontsize = fontsize)
            else:
                plt.title(f'{lcs_id}, {lcs_description}, {s5_type}', fontsize = fontsize)
            plt.xlim((-max_abs,max_abs))
            plt.ylim((-max_abs,max_abs))
            plt.plot(df_isopl_mean, df_isoev_mean,'.', markersize = 3)
            plt.xticks(fontsize = fontsize)
            plt.yticks(fontsize = fontsize)
            plt.xlabel('ISO Pleasantness', fontsize = fontsize)
            plt.ylabel('ISO Eventfulness', fontsize = fontsize)
            plt.grid(visible=True)
            
    def random_choice(self, seed_val = 2023, restrict = False):
        '''
        Picks one random point in time for each of the LCS soundscapes given a random seed value.
        '''
        np.random.seed(seed_val)
        samples = []
        if restrict:
            for prediction_fpath, df in self.paired_predictions_in_quadrants.items():
                df['prediction_fpath'] = prediction_fpath
                samples.append(df.sample())
        else:
            for prediction_fpath, df in self.paired_predictions.items():
                df['prediction_fpath'] = prediction_fpath
                samples.append(df.sample())
        return pd.concat(samples)
    
    
# Note:
# S0003 B&L no points in quadrant
# S0007 B&L no points in quadrant
# S0012 C&R no points in quadrant
# S0018 F&E no points in quadrant
# S0020 B&L no points in quadrant
# S0024 B&L no points in quadrant
# S0027 C&R no points in quadrant
# S0032 F&E no points in quadrant
# S0034 F&E no points in quadrant
# S0038 B&L no points in quadrant
# S0039 B&L no points in quadrant
# S0041 F&E no points in quadrant
# S0048 B&L no points in quadrant
# S0050 B&L no points in quadrant
# S0060 B&L no points in quadrant
# S0061 B&L no points in quadrant