import numpy as np
import scipy.spatial
import pandas as pd

#1234567890123456789012345678901234567890123456789012345678901234
#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#
polar = lambda x, y: (np.sqrt(x**2+y**2), np.arctan2(y, x))
'''
Converts Cartesian coordinates to polar coordinates
print(polar(0,0))
print(polar(-1,0), polar(1,0), polar(0,-1), polar(0,1))
print(polar(3,4), polar(-3,4), polar(3,-4), polar(-3,-4))

Expected output:
(0.0, 0.0)
(1.0, 3.141592653589793) (1.0, 0.0) (1.0, -1.5707963267948966) (1.0, 1.5707963267948966)
(5.0, 0.9272952180016122) (5.0, 2.214297435588181) (5.0, -0.9272952180016122) (5.0, -2.214297435588181)
'''
#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#
def swap_points(df,
                df_dict,
                ConvexHull_weight = 1,
                TDKS_uniform_circle_weight = 1,
                R = 0.5,
                x0 = 0,
                y0 = 0,
                normalise = False,
                permute = 0,
                verbose = 1):
    '''

    ======
    Inputs
    ======
    df : pd.DataFrame (62 rows x 6 columns)
        Assumed to contain the following columns: 
            - start_idxs
            - isopl_1st_half
            - isoev_1st_half
            - isopl_2nd_half
            - isoev_2nd_half
            - prediction_fpath
    df_dict : dict of 62 elements mapping prediction_fpath to a
              pd.DataFrame, each with different numbers of rows
              and 5 columns
        Each dataframe is assumed to contain these columns:
            - start_idxs
            - isopl_1st_half
            - isoev_1st_half
            - isopl_2nd_half
            - isoev_2nd_half
    ConvexHull_weight : float
        The corresponding argument to representative_loss.
    TDKS_uniform_circle_weight : float
        The corresponding argument to representative_loss.
    R : float
        The corresponding argument to TDKS_uniform_circle.
    x0 : float
        The corresponding argument to TDKS_uniform_circle.
    y0 : float
        The corresponding argument to TDKS_uniform_circle.
    normalise : bool
        The corresponding argument to TDKS_uniform_circle.
    permute : bool
        Whether to randomise the order of rows of df before
        making the choice of points to swap, and implement a
        "greedy" algorithm where no more possibilities are
        checked for a given soundscape when an improvement in
        loss function is made.
    verbose : bool
        Whether to print status messages.
        
    ======
    Output
    ======
    best_L : float
        The lowest loss function value resulting from a
        swap of any one row from df with a row with matching
        prediction_fpath (or lcs_id) in df_dict.
    best_df : pd.DataFrame (62 rows x 6 columns)
        The same as df, but with a single row swapped out to
        give the improved loss function value of best_L.
        Will contain the following columns: 
            - start_idxs
            - isopl_1st_half
            - isoev_1st_half
            - isopl_2nd_half
            - isoev_2nd_half
            - prediction_fpath
    '''
    ## INITIALISATION
    correct_dtypes = {'start_idxs'    :  int,
                      'isopl_1st_half':float,
                      'isoev_1st_half':float,
                      'isopl_2nd_half':float,
                      'isoev_2nd_half':float}
        
    x, y = np.concatenate(
        (
            df[['isopl_1st_half','isoev_1st_half']].to_numpy(),
            df[['isopl_2nd_half','isoev_2nd_half']].to_numpy()
        )
    ).T

    L = representative_loss(
        x,y,
        ConvexHull_weight = ConvexHull_weight,
        TDKS_uniform_circle_weight = TDKS_uniform_circle_weight,
        TDKS_uniform_circle_kwargs = {
            'R': R,
            'x0': x0,
            'y0': y0,
            'normalise': normalise,
        }
    )

    if verbose: print(f'Starting loss = {L:.4f}')
    
    ## TAKE INITIAL AS BEST FIRST
    best_L = L
    best_df = df
    
    # RANDOM PERMUTATION OF ROWS HERE IF DESIRED
    if permute:
        df_permuted = df.sample(len(df)) 
    else:
        df_permuted = df

    for idxA, (ridxA, rowA) in enumerate(df_permuted.iterrows()): # Equivalent to going through rows in random order if permute is True
        if verbose: print(f'Currently on row #{idxA+1}/{len(df_permuted)}', end='\r')
        prediction_fpath = rowA['prediction_fpath']
        df_permuted_row_omitted = df_permuted.query('prediction_fpath != @prediction_fpath')
        df_choices = df_dict[prediction_fpath] # Get set that corresponds to candidates to swap
        for ridxB, rowB in df_choices.iterrows():
            df_permuted_row_replaced = pd.concat([df_permuted_row_omitted, rowB.to_frame().T.astype(correct_dtypes)])
            df_permuted_row_replaced['prediction_fpath'].iloc[-1] = prediction_fpath # Can't add to rowB because dtypes are not preserved in .to_frame()
            x, y = np.concatenate(
                (
                    df_permuted_row_replaced[['isopl_1st_half','isoev_1st_half']].to_numpy(),
                    df_permuted_row_replaced[['isopl_2nd_half','isoev_2nd_half']].to_numpy()
                )
            ).T

            L = representative_loss(
                x,y,
                ConvexHull_weight = ConvexHull_weight,
                TDKS_uniform_circle_weight = TDKS_uniform_circle_weight,
                TDKS_uniform_circle_kwargs = {
                    'R': R,
                    'x0': x0, # Mean: 0.10926081. Median: 0.13589937
                    'y0': y0, # Mean: -0.1181002. Median: -0.13430767
                    'normalise': normalise,
                }
            )

            if L < best_L:
                best_L = L
                best_df = df_permuted_row_replaced
                if permute: break
    
    return best_L, best_df
#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#
def quadrant_fractions(x, y, copy = False):
    '''
    For given vectors of points with x-coordinates in x and y-
    coordinates in y, compute the fraction of those vectors with
    (x,y) coordinates lying in the quadrants generated by each
    given vectors (not including the given vector itself).
    
    Vectors with coordinates lying on the boundaries of multiple
    quadrants are counted as lying in all of those quadrants.
    
    ======
    Inputs
    ======
    x : numpy array of shape (n,)
        x-coordinates of input points
    y : numpy array of shape (n,)
        y-coordinates of input points
    copy : bool
        The corresponding argument to np.meshgrid
        
    =======
    Outputs
    =======
    F_top_right : numpy array of shape (n,)
        The fraction of points lying in the top right quadrant
        generated by the points with coordinates in x and y.
    F_top_left : numpy array of shape (n,)
        The fraction of points lying in the top left quadrant
        generated by the points with coordinates in x and y.
    F_bottom_left : numpy array of shape (n,)
        The fraction of points lying in the bottom left quadrant
        generated by the points with coordinates in x and y.
    F_bottom_right : numpy array of shape (n,)
        The fraction of points lying in the bottom right quadrant
        generated by the points with coordinates in x and y.
        
    ========
    Examples
    ========
    x = np.array([0, 0.4,0.6,0.4,0.6, -0.6,-0.4, -0.4,-0.3,-0.2,  0.7])
    y = np.array([0, 0.4,0.4,0.6,0.6,  0.5, 0.5, -0.5,-0.3,-0.5, -0.7])
    plt.figure(figsize=(4,4))
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.plot(x,y,'.',markersize=15)
    plt.grid()

    top_right_observed, top_left_observed, bottom_left_observed, bottom_right_observed = quadrant_fractions(x,y)

    print(top_right_observed)
    print(top_left_observed)
    print(bottom_left_observed)
    print(bottom_right_observed)
    
    # Expected output of the print statements:
    # [0.4        0.25       0.08333333 0.08333333 0.         0.27272727
    #  0.16666667 0.66666667 0.5        0.45454545 0.        ]
    # [0.2        0.25       0.41666667 0.         0.08333333 0.
    #  0.08333333 0.16666667 0.2        0.36363636 1.        ]
    # [0.3        0.33333333 0.41666667 0.58333333 0.75       0.
    #  0.16666667 0.         0.1        0.09090909 0.        ]
    # [0.1        0.16666667 0.08333333 0.33333333 0.16666667 0.72727273
    #  0.58333333 0.16666667 0.2        0.09090909 0.        ]

    ============
    Dependencies
    ============
    numpy (as np)
    
    ===========
    Future work
    ===========
    Consider adding an option such that if a point lies on the boundary of 2 or 4 quadrants up to a certain tolerance $\varepsilon$, assign either 1/2 or 1/4 to each of the quadrants.
    '''
    
    # Error check
    n = x.shape[0]
    if n != y.shape[0]:
        print(f'Warning: Number of x-coordinates ({n}) and y-coordinates ({y.shape[0]}) do not match!')  

    # Make meshgrid to classify points into quadrants
    X, Y = np.meshgrid(x,y,copy=copy)
    left   = X   <= X.T # (i,j)-th element is xi <= xj
    right  = X   >= X.T # (i,j)-th element is xi >= xj
    bottom = Y.T <= Y   # (i,j)-th element is yi <= yj
    top    = Y.T >= Y   # (i,j)-th element is yi >= yj

    # Count number of points in each quadrant, EXCLUDING the point generating the quadrant itself
    # (Hence the minus one in each expression)
    N_top_right    = np.sum(top    & right, axis = 1) - 1
    N_top_left     = np.sum(top    & left , axis = 1) - 1
    N_bottom_left  = np.sum(bottom & left , axis = 1) - 1
    N_bottom_right = np.sum(bottom & right, axis = 1) - 1
    N = (N_top_right + N_top_left + N_bottom_left + N_bottom_right)

    # Normalise to get fraction of points in each quadrant
    F_top_right    = N_top_right    / N
    F_top_left     = N_top_left     / N
    F_bottom_left  = N_bottom_left  / N
    F_bottom_right = N_bottom_right / N
    
    return F_top_right, F_top_left, F_bottom_left, F_bottom_right
#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#
def quadrant_probabilities(x, y, x0 = 0, y0 = 0, R = 1):
    '''
    For given vectors of points with x-coordinates in x and y-
    coordinates in y, compute the probabilities that a uniform
    distribution over a circle with centre (x0, y0) and radius R
    has (x,y) coordinates lying in the quadrants generated by
    the given vectors.
    
    These probabilities are equal to the fractions of the area
    of the circle in each quadrant.
    
    ======
    Inputs
    ======
    x : numpy array of shape (n,)
        x-coordinates of input points
    y : numpy array of shape (n,)
        y-coordinates of input points
    x0 : float
        x-coordinate of circle centre
    y0 : float
        y-coordinate of circle centre
    R : float
        radius of circle
        
    =======
    Outputs
    =======
    A_top_right : numpy array of shape (n,)
        The probability that a point in the circle will lie
        in the top right quadrant generated by the points with
        coordinates in x and y.
    A_top_left : numpy array of shape (n,)
        The probability that a point in the circle will lie
        in the top left quadrant generated by the points with
        coordinates in x and y.
    A_bottom_left : numpy array of shape (n,)
        The probability that a point in the circle will lie
        in the bottom left quadrant generated by the points with
        coordinates in x and y.
    A_bottom_right : numpy array of shape (n,)
        The probability that a point in the circle will lie
        in the bottom right quadrant generated by the points
        with coordinates in x and y.
        
    ========
    Examples
    ========
    x4 = np.array([0.3, -0.3,  0.3, -0.3, 
                   0.4,  0.4, -0.4, -0.4, 
                   1/np.sqrt(2), -1/np.sqrt(2),  1/np.sqrt(2), -1/np.sqrt(2), 
                   0])
    y4 = np.array([0.4,  0.4, -0.4, -0.4, 
                   0.3, -0.3,  0.3, -0.3, 
                   1/np.sqrt(2),  1/np.sqrt(2), -1/np.sqrt(2), -1/np.sqrt(2), 
                   0])
    x3 = np.array([0.7, -0.7,  0.7, -0.7, 
                   0.9,  0.9, -0.9, -0.9])
    y3 = np.array([0.9,  0.9, -0.9, -0.9, 
                   0.7, -0.7,  0.7, -0.7])
    x2 = np.array([ 0,    0,  0.3,  0.3, -0.4, -0.4, 
                    1,    1,    1,  1.2,  1.2,  1.2, 
                    0,    0,  0.2,  0.2, -0.6, -0.6,
                   -1,   -1,   -1, -1.2, -1.2, -1.2,])
    y2 = np.array([ 1,  1.2,    1,  1.2,    1,  1.2, 
                    0,  0.8, -0.2,    0,  0.8, -0.2, 
                   -1, -1.2,   -1, -1.2,   -1, -1.2,
                    0,  0.7, -0.3,    0,  0.7, -0.3,])
    x1 = np.array([1,   1, 1.2, -1,  -1, -1.2,  1,    1, 1.2, -1,   -1, -1.2])
    y1 = np.array([1, 1.2,   1,  1, 1.2,    1, -1, -1.2,  -1, -1, -1.2,   -1])

    x = np.concatenate((x4, x3, x2, x1))
    y = np.concatenate((y4, y3, y2, y1))

    a,b,c,d = quadrant_probabilities(x, y)

    theta = np.linspace(0,2*np.pi,1000)
    x_circ, y_circ = np.cos(theta), np.sin(theta)
    plt.figure(figsize=(6,6))
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)
    plt.plot(x_circ, y_circ, 'b')
    plt.plot(x4, y4, 'y.', markersize = 15)
    plt.plot(x3, y3, 'go', markersize = 10, fillstyle='none')
    plt.plot(x2, y2, 'k^', markersize = 10)
    plt.plot(x1, y1, 'c*', markersize = 10)
    plt.grid()

    for i in [a,b,c,d]:
        print(i)
    print(a+b+c+d)

    # Expected output of the print statements:
    # [7.03144964e-02 1.82001291e-01 2.41604336e-01 5.06079876e-01
    #  7.03144964e-02 1.82001291e-01 2.41604336e-01 5.06079876e-01
    #  1.96173613e-33 9.08450569e-02 9.08450569e-02 8.18309886e-01
    #  2.50000000e-01 0.00000000e+00 4.53809089e-02 1.54997927e-01
    #  7.99621164e-01 0.00000000e+00 4.53809089e-02 1.54997927e-01
    #  7.99621164e-01 0.00000000e+00 0.00000000e+00 0.00000000e+00
    #  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
    #  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
    #  0.00000000e+00 5.00000000e-01 5.00000000e-01 3.73530039e-01
    #  3.73530039e-01 8.57621510e-01 8.57621510e-01 5.00000000e-01
    #  9.40602022e-02 6.88081168e-01 5.00000000e-01 9.40602022e-02
    #  6.88081168e-01 0.00000000e+00 0.00000000e+00 0.00000000e+00
    #  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
    #  0.00000000e+00 0.00000000e+00 1.00000000e+00 1.00000000e+00
    #  1.00000000e+00]
    # [1.82001291e-01 7.03144964e-02 5.06079876e-01 2.41604336e-01
    #  2.41604336e-01 5.06079876e-01 7.03144964e-02 1.82001291e-01
    #  9.08450569e-02 1.96173613e-33 8.18309886e-01 9.08450569e-02
    #  2.50000000e-01 4.53809089e-02 0.00000000e+00 7.99621164e-01
    #  1.54997927e-01 1.54997927e-01 7.99621164e-01 0.00000000e+00
    #  4.53809089e-02 0.00000000e+00 0.00000000e+00 0.00000000e+00
    #  0.00000000e+00 0.00000000e+00 0.00000000e+00 5.00000000e-01
    #  5.20440193e-02 6.26469961e-01 5.00000000e-01 5.20440193e-02
    #  6.26469961e-01 5.00000000e-01 5.00000000e-01 6.26469961e-01
    #  6.26469961e-01 1.42378490e-01 1.42378490e-01 0.00000000e+00
    #  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
    #  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
    #  0.00000000e+00 0.00000000e+00 0.00000000e+00 1.00000000e+00
    #  1.00000000e+00 1.00000000e+00 0.00000000e+00 0.00000000e+00
    #  0.00000000e+00]
    # [5.06079876e-01 2.41604336e-01 1.82001291e-01 7.03144964e-02
    #  5.06079876e-01 2.41604336e-01 1.82001291e-01 7.03144964e-02
    #  8.18309886e-01 9.08450569e-02 9.08450569e-02 1.96173613e-33
    #  2.50000000e-01 7.99621164e-01 1.54997927e-01 4.53809089e-02
    #  0.00000000e+00 7.99621164e-01 1.54997927e-01 4.53809089e-02
    #  0.00000000e+00 5.00000000e-01 5.00000000e-01 6.88081168e-01
    #  6.88081168e-01 2.52315788e-01 2.52315788e-01 5.00000000e-01
    #  9.47955981e-01 3.73530039e-01 5.00000000e-01 9.47955981e-01
    #  3.73530039e-01 0.00000000e+00 0.00000000e+00 0.00000000e+00
    #  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
    #  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
    #  0.00000000e+00 1.00000000e+00 1.00000000e+00 1.00000000e+00
    #  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
    #  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
    #  0.00000000e+00]
    # [2.41604336e-01 5.06079876e-01 7.03144964e-02 1.82001291e-01
    #  1.82001291e-01 7.03144964e-02 5.06079876e-01 2.41604336e-01
    #  9.08450569e-02 8.18309886e-01 1.96173613e-33 9.08450569e-02
    #  2.50000000e-01 1.54997927e-01 7.99621164e-01 0.00000000e+00
    #  4.53809089e-02 4.53809089e-02 0.00000000e+00 7.99621164e-01
    #  1.54997927e-01 5.00000000e-01 5.00000000e-01 3.11918832e-01
    #  3.11918832e-01 7.47684212e-01 7.47684212e-01 0.00000000e+00
    #  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
    #  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
    #  0.00000000e+00 0.00000000e+00 0.00000000e+00 5.00000000e-01
    #  9.05939798e-01 3.11918832e-01 5.00000000e-01 9.05939798e-01
    #  3.11918832e-01 0.00000000e+00 0.00000000e+00 0.00000000e+00
    #  1.00000000e+00 1.00000000e+00 1.00000000e+00 0.00000000e+00
    #  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
    #  0.00000000e+00]
    # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
    #  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
    #  1. 1. 1. 1. 1. 1. 1. 1. 1.]

    ============
    Dependencies
    ============
    numpy (as np), split_quads, quadrant_probabilities_1,
    quadrant_probabilities_2, quadrant_probabilities_3,
    quadrant_probabilities_4
    '''
    # Error check
    n = x.shape[0]
    if n != y.shape[0]:
        print(f'Warning: Number of x-coordinates ({n}) and y-coordinates ({y.shape[0]}) do not match!')    
    
    # Compute probabilities for all quadrants in four possible cases: where the point splits the circle into 1, 2, 3, and 4 quadrants with non-zero area.
    idxs_4quad, idxs_3quad, idxs_2quad, idxs_1quad = split_quads(x, y, x0 = x0, y0 = y0, R = R, return_type = 'indices')

    # Preallocate output arrays
    A_top_right    = np.zeros(n)
    A_top_left     = np.zeros(n)
    A_bottom_left  = np.zeros(n)
    A_bottom_right = np.zeros(n)
    
    # Assign elements to output arrays by case
    for idxs_kquad, quadrant_probabilities_k in zip([idxs_1quad, idxs_2quad, idxs_3quad, idxs_4quad],
                                                    [quadrant_probabilities_1, quadrant_probabilities_2, quadrant_probabilities_3, quadrant_probabilities_4]):
        A_top_right[idxs_kquad], A_top_left[idxs_kquad], A_bottom_left[idxs_kquad], A_bottom_right[idxs_kquad] = quadrant_probabilities_k(x[idxs_kquad], y[idxs_kquad], x0 = x0, y0 = y0, R = R)
        
    return A_top_right, A_top_left, A_bottom_left, A_bottom_right
#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#
def quadrant_probabilities_1(x, y, x0 = 0, y0 = 0, R = 1):
    '''
    For given vectors of points with x-coordinates in x and y-
    coordinates in y, compute the probabilities that a uniform
    distribution over a circle with centre (x0, y0) and radius R
    has (x,y) coordinates lying in the quadrants generated by
    the given vectors, PROVIDED that the points have both x and
    y coordinate magnitudes exceeding the circle radius
    
    These probabilities are equal to the fractions of the area
    of the circle in each quadrant.
    
    ======
    Inputs
    ======
    x : numpy array of shape (n,)
        x-coordinates of input points
    y : numpy array of shape (n,)
        y-coordinates of input points
    x0 : float
        x-coordinate of circle centre
    y0 : float
        y-coordinate of circle centre
    R : float
        radius of circle
        
    =======
    Outputs
    =======
    A_top_right : numpy array of shape (n,)
        The probability that a point in the circle will lie
        in the top right quadrant generated by the points with
        coordinates in x and y.
    A_top_left : numpy array of shape (n,)
        The probability that a point in the circle will lie
        in the top left quadrant generated by the points with
        coordinates in x and y.
    A_bottom_left : numpy array of shape (n,)
        The probability that a point in the circle will lie
        in the bottom left quadrant generated by the points with
        coordinates in x and y.
    A_bottom_right : numpy array of shape (n,)
        The probability that a point in the circle will lie
        in the bottom right quadrant generated by the points
        with coordinates in x and y.
        
    ========
    Examples
    ========
    x = np.array([1,   1, 1.2, -1,  -1, -1.2,  1,    1, 1.2, -1,   -1, -1.2])
    y = np.array([1, 1.2,   1,  1, 1.2,    1, -1, -1.2,  -1, -1, -1.2,   -1])
    a, b, c, d = quadrant_probabilities_1(x,y)
    print(a)
    print(b)
    print(c)
    print(d)
    print(a+b+c+d)

    theta = np.linspace(0, 2*np.pi, 1000)
    x_circ, y_circ = np.cos(theta), np.sin(theta)
    plt.figure(figsize=(4,4))
    plt.plot(x_circ, y_circ, 'b')
    plt.plot(x,y,'k.',markersize=10)
    plt.grid()
    
    # Expected output of the print statements:
    # [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1.]
    # [0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 0. 0.]
    # [1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    # [0. 0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0.]
    # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
    
    ============
    Dependencies
    ============
    numpy (as np)
    '''
    # Centre all points first
    x = x - x0
    y = y - y0
    
    # Error check
    if not np.all((np.abs(x) >= R) & (np.abs(y) >= R)):
        print('Warning: quadrant_probabilities_1 assumes all points have BOTH x- and y-coordinates with magnitude larger than the circle radius.')
        print('However, not all of the given points satisfy this condition. Answers may wrongly involve areas not summing to 1.')
    if R <= 0:
        print('Warning: Radius of circle R is 0 or negative. Answers may wrongly involve areas not summing to 1.')
    
    # Get the probabilities as total areas (segment + triangle), divided by the area of the circle
    A_top_right    = ((x < 0) & (y < 0)).astype(float)
    A_top_left     = ((x > 0) & (y < 0)).astype(float)
    A_bottom_left  = ((x > 0) & (y > 0)).astype(float)
    A_bottom_right = ((x < 0) & (y > 0)).astype(float)
    
    return A_top_right, A_top_left, A_bottom_left, A_bottom_right
#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#
def quadrant_probabilities_2(x, y, x0 = 0, y0 = 0, R = 1):
    '''
    For given vectors of points with x-coordinates in x and y-
    coordinates in y, compute the probabilities that a uniform
    distribution over a circle with centre (x0, y0) and radius R
    has (x,y) coordinates lying in the quadrants generated by
    the given vectors, PROVIDED that exactly one coordinate has
    magnitude larger than R.
    
    These probabilities are equal to the fractions of the area
    of the circle in each quadrant.
    
    ======
    Inputs
    ======
    x : numpy array of shape (n,)
        x-coordinates of input points
    y : numpy array of shape (n,)
        y-coordinates of input points
    x0 : float
        x-coordinate of circle centre
    y0 : float
        y-coordinate of circle centre
    R : float
        radius of circle
        
    =======
    Outputs
    =======
    A_top_right : numpy array of shape (n,)
        The probability that a point in the circle will lie
        in the top right quadrant generated by the points with
        coordinates in x and y.
    A_top_left : numpy array of shape (n,)
        The probability that a point in the circle will lie
        in the top left quadrant generated by the points with
        coordinates in x and y.
    A_bottom_left : numpy array of shape (n,)
        The probability that a point in the circle will lie
        in the bottom left quadrant generated by the points with
        coordinates in x and y.
    A_bottom_right : numpy array of shape (n,)
        The probability that a point in the circle will lie
        in the bottom right quadrant generated by the points
        with coordinates in x and y.
        
    ========
    Examples
    ========
    x = np.array([ 0,    0,  0.3,  0.3, -0.4, -0.4, 
                   1,    1,    1,  1.2,  1.2,  1.2, 
                   0,    0,  0.2,  0.2, -0.6, -0.6,
                  -1,   -1,   -1, -1.2, -1.2, -1.2,])
    y = np.array([ 1,  1.2,    1,  1.2,    1,  1.2, 
                   0,  0.8, -0.2,    0,  0.8, -0.2, 
                  -1, -1.2,   -1, -1.2,   -1, -1.2,
                   0,  0.7, -0.3,    0,  0.7, -0.3,])
    a, b, c, d = quadrant_probabilities_2(x,y)
    for start_idx, end_idx in zip(range(0,19,6), range(6,25,6)):
        print(a[start_idx:end_idx])
        print(b[start_idx:end_idx])
        print(c[start_idx:end_idx])
        print(d[start_idx:end_idx])
        print((a+b+c+d)[start_idx:end_idx])
        print()

    x_bad = np.array([1,   1, 1.2, -1,  -1, -1.2,  1,    1, 1.2, -1,   -1, -1.2])
    y_bad = np.array([1, 1.2,   1,  1, 1.2,    1, -1, -1.2,  -1, -1, -1.2,   -1])
    a, b, c, d = quadrant_probabilities_2(x_bad,y_bad)
    print(a)
    print(b)
    print(c)
    print(d)
    print(a+b+c+d)

    theta = np.linspace(0, 2*np.pi, 1000)
    x_circ, y_circ = np.cos(theta), np.sin(theta)
    plt.figure(figsize=(4,4))
    plt.plot(x_circ, y_circ, 'b')
    plt.plot(x,y,'k.',markersize=10)
    plt.plot(x_bad,y_bad,'ro',markersize=10,fillstyle='none')
    plt.grid()
    
    # Expected output of the print statements:
    # [0. 0. 0. 0. 0. 0.]
    # [0. 0. 0. 0. 0. 0.]
    # [0.5        0.5        0.68808117 0.68808117 0.25231579 0.25231579]
    # [0.5        0.5        0.31191883 0.31191883 0.74768421 0.74768421]
    # [1. 1. 1. 1. 1. 1.]
    # 
    # [0. 0. 0. 0. 0. 0.]
    # [0.5        0.05204402 0.62646996 0.5        0.05204402 0.62646996]
    # [0.5        0.94795598 0.37353004 0.5        0.94795598 0.37353004]
    # [0. 0. 0. 0. 0. 0.]
    # [1. 1. 1. 1. 1. 1.]
    # 
    # [0.5        0.5        0.37353004 0.37353004 0.85762151 0.85762151]
    # [0.5        0.5        0.62646996 0.62646996 0.14237849 0.14237849]
    # [0. 0. 0. 0. 0. 0.]
    # [0. 0. 0. 0. 0. 0.]
    # [1. 1. 1. 1. 1. 1.]
    # 
    # [0.5        0.0940602  0.68808117 0.5        0.0940602  0.68808117]
    # [0. 0. 0. 0. 0. 0.]
    # [0. 0. 0. 0. 0. 0.]
    # [0.5        0.9059398  0.31191883 0.5        0.9059398  0.31191883]
    # [1. 1. 1. 1. 1. 1.]
    # 
    # Warning: quadrant_probabilities_2 assumes all points have exactly one (and not both) coordinates with magnitude larger than R = 1.
    # However, not all of the given points satisfy this condition. Answers may wrongly involve negative numbers or nans.
    # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    
    ============
    Dependencies
    ============
    numpy (as np)
    '''
    # Centre all points first
    x = x - x0
    y = y - y0
    
    # Error check
    x_coords_below_R = np.abs(x) < R
    y_coords_below_R = np.abs(y) < R
    if not np.all(x_coords_below_R ^ y_coords_below_R):
        print(f'Warning: quadrant_probabilities_2 assumes all points have exactly one (and not both) coordinates with magnitude larger than R = {R}.')
        print('However, not all of the given points satisfy this condition. Answers may wrongly involve negative numbers or nans.')
    if R <= 0:
        print('Warning: Radius of circle R is 0 or negative. Answers may wrongly involve areas not summing to 1.')
    
    # Get Cartesian coordinates of the intersection points, if any (if not, set coordinates to 0 to prevent errors with np.sqrt)
    # The corresponding intersection points exist iff R**2 - y**2 >= 0 or R**2 - x**2 >= 0
    xL, xR = -np.sqrt(np.maximum(R**2 - y**2, 0)), np.sqrt(np.maximum(R**2 - y**2, 0))
    yB, yT = -np.sqrt(np.maximum(R**2 - x**2, 0)), np.sqrt(np.maximum(R**2 - x**2, 0))
    
    # Get polar coordinates of the intersection points (only angle is required)
    thetaR = np.arctan2(y , xR)
    thetaT = np.arctan2(yT, x )
    thetaL = np.arctan2(y , xL)
    thetaB = np.arctan2(yB, x )
    
    # Get the directed angles of the subtended arcs
    thetaTB = thetaT-thetaB
    thetaLR = (thetaL-thetaR)%(2*np.pi) # Remainder needed to correct direction due to interval of arctan2
    
    # Get the segment areas
    AsTB = R*R*(thetaTB - np.sin(thetaTB))/2 # RHS
    AsLR = R*R*(thetaLR - np.sin(thetaLR))/2 # TOP
    
    # Get the probabilities as one of four cases depending on which pair of quadrants is empty
    A_top_right    = ((x_coords_below_R & (y > 0))*(               0. ) + (x_coords_below_R & (y < 0))*(             AsTB ) + (y_coords_below_R & (x < 0))*(             AsLR ) + (y_coords_below_R & (x > 0))*(               0. ))/np.pi/R/R
    A_top_left     = ((x_coords_below_R & (y > 0))*(               0. ) + (x_coords_below_R & (y < 0))*( R*R*np.pi - AsTB ) + (y_coords_below_R & (x < 0))*(               0. ) + (y_coords_below_R & (x > 0))*(             AsLR ))/np.pi/R/R
    A_bottom_left  = ((x_coords_below_R & (y > 0))*( R*R*np.pi - AsTB ) + (x_coords_below_R & (y < 0))*(               0. ) + (y_coords_below_R & (x < 0))*(               0. ) + (y_coords_below_R & (x > 0))*( R*R*np.pi - AsLR ))/np.pi/R/R
    A_bottom_right = ((x_coords_below_R & (y > 0))*(             AsTB ) + (x_coords_below_R & (y < 0))*(               0. ) + (y_coords_below_R & (x < 0))*( R*R*np.pi - AsLR ) + (y_coords_below_R & (x > 0))*(               0. ))/np.pi/R/R
    
    #return thetaTB, thetaLR, AsTB, AsLR
    return A_top_right, A_top_left, A_bottom_left, A_bottom_right
#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#
def quadrant_probabilities_3(x, y, x0 = 0, y0 = 0, R = 1):
    '''
    For given vectors of points with x-coordinates in x and y-
    coordinates in y, compute the probabilities that a uniform
    distribution over a circle with centre (x0, y0) and radius R
    has (x,y) coordinates lying in the quadrants generated by
    the given vectors, PROVIDED that the points all lie outside
    the circle but not the square circumscribing the circle.
    
    These probabilities are equal to the fractions of the area
    of the circle in each quadrant.
    
    ======
    Inputs
    ======
    x : numpy array of shape (n,)
        x-coordinates of input points
    y : numpy array of shape (n,)
        y-coordinates of input points
    x0 : float
        x-coordinate of circle centre
    y0 : float
        y-coordinate of circle centre
    R : float
        radius of circle
        
    =======
    Outputs
    =======
    A_top_right : numpy array of shape (n,)
        The probability that a point in the circle will lie
        in the top right quadrant generated by the points with
        coordinates in x and y.
    A_top_left : numpy array of shape (n,)
        The probability that a point in the circle will lie
        in the top left quadrant generated by the points with
        coordinates in x and y.
    A_bottom_left : numpy array of shape (n,)
        The probability that a point in the circle will lie
        in the bottom left quadrant generated by the points with
        coordinates in x and y.
    A_bottom_right : numpy array of shape (n,)
        The probability that a point in the circle will lie
        in the bottom right quadrant generated by the points
        with coordinates in x and y.
        
    ========
    Examples
    ========
    x = np.array([0.7, -0.7,  0.7, -0.7])
    y = np.array([0.9,  0.9, -0.9, -0.9])
    a, b, c, d = quadrant_probabilities_3(x,y)
    print(a)
    print(b)
    print(c)
    print(d)
    print(a+b+c+d)

    x_bad = np.array([1, 0.3])
    y_bad = np.array([0, 0.4])
    a, b, c, d = quadrant_probabilities_3(x_bad,y_bad)
    print(a)
    print(b)
    print(c)
    print(d)
    print(a+b+c+d)

    theta = np.linspace(0, 2*np.pi, 1000)
    x_circ, y_circ = np.cos(theta), np.sin(theta)
    plt.figure(figsize=(4,4))
    plt.plot(x_circ, y_circ, 'b')
    plt.plot(x,y,'k.',markersize=10)
    plt.plot(x_bad,y_bad,'ro',markersize=10,fillstyle='none')
    plt.grid()

    # Expected output of the print statements:
    # [0.         0.04538091 0.15499793 0.79962116]
    # [0.04538091 0.         0.79962116 0.15499793]
    # [0.79962116 0.15499793 0.04538091 0.        ]
    # [0.15499793 0.79962116 0.         0.04538091]
    # [1. 1. 1. 1.]
    # Warning: quadrant_probabilities_3 assumes all points lie outside the circle but not the square circumscribing the circle.
    # However, not all of the given points satisfy this condition. Answers may wrongly involve negative numbers or nans.
    # [0. 0.]
    # [0.         0.07475069]
    # [0.         0.81649543]
    # [0.         0.10875388]
    # [0. 1.]

    ============
    Dependencies
    ============
    numpy (as np)
    '''
    # Centre all points first
    x = x - x0
    y = y - y0
    
    # Error check
    if not np.all((x**2 + y**2 >= R**2) & (np.abs(x) < R) & (np.abs(y) < R)):
        print('Warning: quadrant_probabilities_3 assumes all points lie outside the circle but not the square circumscribing the circle.')
        print('However, not all of the given points satisfy this condition. Answers may wrongly involve negative numbers or nans.')

    # Get Cartesian coordinates of the intersection points
    xL, xR = -np.sqrt(R**2 - y**2), np.sqrt(R**2 - y**2) 
    yB, yT = -np.sqrt(R**2 - x**2), np.sqrt(R**2 - x**2)
    
    # Get polar coordinates of the intersection points (only angle is required)
    thetaR = np.arctan2(y , xR)
    thetaT = np.arctan2(yT, x )
    thetaL = np.arctan2(y , xL)
    thetaB = np.arctan2(yB, x )
    
    # Get the angles of the subtended arcs
    thetaTR = np.minimum( np.abs(thetaR-thetaT), 2*np.pi-np.abs(thetaR-thetaT) )
    thetaTL = np.minimum( np.abs(thetaT-thetaL), 2*np.pi-np.abs(thetaT-thetaL) )
    thetaBL = np.minimum( np.abs(thetaL-thetaB), 2*np.pi-np.abs(thetaL-thetaB) )
    thetaBR = np.minimum( np.abs(thetaB-thetaR), 2*np.pi-np.abs(thetaB-thetaR) )
    
    # Get the segment areas
    AsTR = R*R*(thetaTR - np.sin(thetaTR))/2
    AsTL = R*R*(thetaTL - np.sin(thetaTL))/2
    AsBL = R*R*(thetaBL - np.sin(thetaBL))/2
    AsBR = R*R*(thetaBR - np.sin(thetaBR))/2
    
    # Get the probabilities as one of four cases depending on which quadrant is empty
    A_top_right    = ( ((x > 0) & (y > 0))*( 0. ) + (((x < 0) & (y > 0)) | ((x > 0) & (y < 0)))*( AsTR ) + ((x < 0) & (y < 0))*( np.pi*R*R - AsTL - AsBR ) )/np.pi/R/R
    A_top_left     = ( ((x < 0) & (y > 0))*( 0. ) + (((x < 0) & (y < 0)) | ((x > 0) & (y > 0)))*( AsTL ) + ((x > 0) & (y < 0))*( np.pi*R*R - AsTR - AsBL ) )/np.pi/R/R
    A_bottom_left  = ( ((x < 0) & (y < 0))*( 0. ) + (((x > 0) & (y < 0)) | ((x < 0) & (y > 0)))*( AsBL ) + ((x > 0) & (y > 0))*( np.pi*R*R - AsTL - AsBR ) )/np.pi/R/R
    A_bottom_right = ( ((x > 0) & (y < 0))*( 0. ) + (((x > 0) & (y > 0)) | ((x < 0) & (y < 0)))*( AsBR ) + ((x < 0) & (y > 0))*( np.pi*R*R - AsTR - AsBL ) )/np.pi/R/R
    
    return A_top_right, A_top_left, A_bottom_left, A_bottom_right
#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#
def quadrant_probabilities_4(x, y, x0 = 0, y0 = 0, R = 1):
    '''
    For given vectors of points with x-coordinates in x and y-
    coordinates in y, compute the probabilities that a uniform
    distribution over a circle with centre (x0, y0) and radius R
    has (x,y) coordinates lying in the quadrants generated by
    the given vectors, PROVIDED that the points all lie within
    the circle.
    
    These probabilities are equal to the fractions of the area
    of the circle in each quadrant.
    
    ======
    Inputs
    ======
    x : numpy array of shape (n,)
        x-coordinates of input points
    y : numpy array of shape (n,)
        y-coordinates of input points
    x0 : float
        x-coordinate of circle centre
    y0 : float
        y-coordinate of circle centre
    R : float
        radius of circle
        
    =======
    Outputs
    =======
    A_top_right : numpy array of shape (n,)
        The probability that a point in the circle will lie
        in the top right quadrant generated by the points with
        coordinates in x and y.
    A_top_left : numpy array of shape (n,)
        The probability that a point in the circle will lie
        in the top left quadrant generated by the points with
        coordinates in x and y.
    A_bottom_left : numpy array of shape (n,)
        The probability that a point in the circle will lie
        in the bottom left quadrant generated by the points with
        coordinates in x and y.
    A_bottom_right : numpy array of shape (n,)
        The probability that a point in the circle will lie
        in the bottom right quadrant generated by the points
        with coordinates in x and y.
        
    ========
    Examples
    ========
    x = np.array([0.3, -0.3,  0.3, -0.3])
    y = np.array([0.4,  0.4, -0.4, -0.4])
    a, b, c, d = quadrant_probabilities_4(x,y)
    print(a)
    print(b)
    print(c)
    print(d)
    print(a+b+c+d)

    theta = np.linspace(0, 2*np.pi, 1000)
    x_circ, y_circ = np.cos(theta), np.sin(theta)
    plt.figure(figsize=(4,4))
    plt.plot(x_circ, y_circ, 'b')
    plt.plot(x,y,'k.',markersize=10)
    plt.grid()

    x_bad = np.array([1, 0.9])
    y_bad = np.array([0, 0.9])
    a, b, c, d = quadrant_probabilities_4(x_bad,y_bad)
    print(a)
    print(b)
    print(c)
    print(d)
    print(a+b+c+d)
    
    # Expected output of the print statements:
    # [0.0703145  0.18200129 0.24160434 0.50607988]
    # [0.18200129 0.0703145  0.50607988 0.24160434]
    # [0.50607988 0.24160434 0.18200129 0.0703145 ]
    # [0.24160434 0.50607988 0.0703145  0.18200129]
    # [1. 1. 1. 1.]
    # Warning: quadrant_probabilities_4 assumes all points lie within the circle but not all of the given points do. Answers may wrongly involve negative numbers or nans.
    # [0.         0.04203933]
    # [ 0.5        -0.00783101]
    # [0.5        0.57891843]
    # [ 0.         -0.00783101]
    # [1.         0.60529574]
    
    ============
    Dependencies
    ============
    numpy (as np)
    '''
    # Centre all points first
    x = x - x0
    y = y - y0
    
    # Error check
    if not np.all(x**2 + y**2 < R**2):
        print('Warning: quadrant_probabilities_4 assumes all points lie within the circle but not all of the given points do. Answers may wrongly involve negative numbers or nans.')
    
    # Get Cartesian coordinates of the intersection points
    xL, xR = -np.sqrt(R**2 - y**2), np.sqrt(R**2 - y**2)
    yB, yT = -np.sqrt(R**2 - x**2), np.sqrt(R**2 - x**2)
    
    # Get the triangle areas
    AtTR = (xR-x )*(yT-y )/2
    AtTL = (x -xL)*(yT-y )/2
    AtBL = (x -xL)*(y -yB)/2
    AtBR = (xR-x )*(y -yB)/2
    
    # Get polar coordinates of the intersection points (only angle is required)
    thetaR = np.arctan2(y , xR)
    thetaT = np.arctan2(yT, x )
    thetaL = np.arctan2(y , xL)
    thetaB = np.arctan2(yB, x )
    
    # Get the angles of the subtended arcs
    thetaTR = np.minimum( np.abs(thetaR-thetaT), 2*np.pi-np.abs(thetaR-thetaT) )
    thetaTL = np.minimum( np.abs(thetaT-thetaL), 2*np.pi-np.abs(thetaT-thetaL) )
    thetaBL = np.minimum( np.abs(thetaL-thetaB), 2*np.pi-np.abs(thetaL-thetaB) )
    thetaBR = np.minimum( np.abs(thetaB-thetaR), 2*np.pi-np.abs(thetaB-thetaR) )
    
    # Get the segment areas
    AsTR = R*R*(thetaTR - np.sin(thetaTR))/2
    AsTL = R*R*(thetaTL - np.sin(thetaTL))/2
    AsBL = R*R*(thetaBL - np.sin(thetaBL))/2
    AsBR = R*R*(thetaBR - np.sin(thetaBR))/2
    
    # Get the probabilities as total areas (segment + triangle), divided by the area of the circle
    A_top_right    = (AsTR + AtTR)/np.pi/R/R
    A_top_left     = (AsTL + AtTL)/np.pi/R/R
    A_bottom_left  = (AsBL + AtBL)/np.pi/R/R
    A_bottom_right = (AsBR + AtBR)/np.pi/R/R
    
    return A_top_right, A_top_left, A_bottom_left, A_bottom_right
#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#
def representative_loss(x, y,
                        ConvexHull_weight = 1,
                        TDKS_uniform_circle_weight = 1,
                        ConvexHull_kwargs = {},
                        TDKS_uniform_circle_kwargs = {}):
    '''
    For given vectors of points with x-coordinates in x and y-
    coordinates in y, compute the representative loss function
    for the Lion City Soundscapes dataset. This is a weighted
    sum of the negative volume of the convex hull and the
    2-dimensional Kolmogorov-Smirnov test statistic of the set
    of points.
    
    ======
    Inputs
    ======
    x : numpy array of shape (n,)
        x-coordinates of input points
    y : numpy array of shape (n,)
        y-coordinates of input points
    ConvexHull_weight : float
        The weight of the convex hull term in in the loss
        function.
    TDKS_uniform_circle_weight : float
        The weight of the 2-dimensional Kolmogorov-Smirnov
        test statistic in the loss function.
    ConvexHull_kwargs : dict
        Keyword arguments to scipy.spatial.ConvexHull.
    TDKS_uniform_circle_kwargs : dict
        Keyword arguments to TDKS_uniform_circle.
    
    =======
    Outputs
    =======
    L_r : float
        The value of the representative loss function for
        the input points.
    
    ============
    Dependencies
    ============
    scipy.spatial.ConvexHull, numpy (as np),
    TDKS_uniform_circle

    '''
    # Error check:
    if ConvexHull_weight < 0:
        print(f'Warning: ConvexHull_weight must be non-negative but got {ConvexHull_weight} instead.')    
    if TDKS_uniform_circle_weight < 0:
        print(f'Warning: TDKS_uniform_circle_weight must be non-negative but got {TDKS_uniform_circle_weight} instead.')
    
    # Computation of loss function    
    CH = scipy.spatial.ConvexHull(np.stack((x,y)).T, **ConvexHull_kwargs)
    L_r = - ConvexHull_weight*CH.volume + TDKS_uniform_circle_weight*TDKS_uniform_circle(x,y, **TDKS_uniform_circle_kwargs)
    return L_r
#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#
def split_quads(x, y, x0 = 0, y0 = 0, R = 1, return_type = 'points'):
    '''
    For n given vectors of points with x-coordinates in x and y-
    coordinates in y, separate the points into four distinct
    sets where each set of points splits the PDF of the 
    uniform distribution over a circle with centre (x0, y0) and
    radius R into the same number of regions with non-zero area
    when a vertical and horizontal line is drawn at the
    coordinates (x,y). The sets have cardinality n4, n3, n2, and
    n1, where n1 + n2 + n3 + n4 = n.
    
    ======
    Inputs
    ======
    x : numpy array of shape (n,)
        x-coordinates of input points
    y : numpy array of shape (n,)
        y-coordinates of input points
    x0 : float
        x-coordinate of circle centre
    y0 : float
        y-coordinate of circle centre
    R : float
        radius of circle
    return_type : str in ['points','indices']
        if 'points', returns x_#quad, y_#quad for # in {1,2,3,4}
        if 'indices', returns 
    
    =======
    Outputs
    =======
    x_4quad : numpy array of shape (n4,)
        x-coordinates of input points that split the circle into 4 non-zero regions
    y_4quad : numpy array of shape (n4,)
        y-coordinates of input points that split the circle into 4 non-zero regions
    x_3quad : numpy array of shape (n3,)
        x-coordinates of input points that split the circle into 3 non-zero regions
    y_3quad : numpy array of shape (n3,)
        y-coordinates of input points that split the circle into 3 non-zero regions
    x_2quad : numpy array of shape (n2,)
        x-coordinates of input points that split the circle into 2 non-zero regions
    y_2quad : numpy array of shape (n2,)
        y-coordinates of input points that split the circle into 2 non-zero regions
    x_1quad : numpy array of shape (n1,)
        x-coordinates of input points that split the circle into 1 non-zero region (i.e., the entire circle lies in one quadrant)
    y_1quad : numpy array of shape (n1,)
        y-coordinates of input points that split the circle into 1 non-zero region (i.e., the entire circle lies in one quadrant)
        
    idxs_4quad : numpy array of shape (n,)
        Boolean array where the i-th element is True iff (x[i],y[i]) splits the circle into 4 non-zero regions (n4 such elements)
    idxs_3quad : numpy array of shape (n,)
        Boolean array where the i-th element is True iff (x[i],y[i]) splits the circle into 3 non-zero regions (n3 such elements)
    idxs_2quad : numpy array of shape (n,)
        Boolean array where the i-th element is True iff (x[i],y[i]) splits the circle into 2 non-zero regions (n2 such elements)
    idxs_1quad : numpy array of shape (n,)
        Boolean array where the i-th element is True iff (x[i],y[i]) splits the circle into 1 non-zero regions (n1 such elements)
    
    ========
    Examples
    ========
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Do a rough colouring of the regions in four colours.
    x = np.random.rand(100000)*2 - 1
    y = np.random.rand(100000)*2 - 1
    x_4quad, y_4quad, x_3quad, y_3quad, x_2quad, y_2quad, x_1quad, y_1quad = split_quads(x, y, R = 0.5)
    plt.figure(figsize=(6,6))
    plt.plot(x_4quad, y_4quad, 'y.', markersize = 3)
    plt.plot(x_3quad, y_3quad, 'g.', markersize = 3)
    plt.plot(x_2quad, y_2quad, 'k.', markersize = 3)
    plt.plot(x_1quad, y_1quad, 'c.', markersize = 3)
    plt.grid()
    
    # Show some sample and special cases.
    x = np.array([0, -0.5, -1/np.sqrt(2), np.cos(np.pi/4), 0.8, -0.8,  0.8, -0.8, 0.8,  -1,  1.5, 0, 1, -1,  1, -1, 1.5])
    y = np.array([0,  0.5, -1/np.sqrt(2), np.sin(np.pi/4), 0.8,  0.8, -0.8, -0.8,   1, 0.8, -0.5, 1, 1,  1, -1, -1, 1.5])
    theta = np.linspace(0,2*np.pi,1000)
    x_circ, y_circ = np.cos(theta), np.sin(theta)
    x_4quad, y_4quad, x_3quad, y_3quad, x_2quad, y_2quad, x_1quad, y_1quad = split_quads(x, y)
    plt.figure(figsize=(6,6))
    plt.plot(x_circ, y_circ, 'b')
    plt.plot(x_4quad, y_4quad, 'r.', markersize = 15)
    plt.plot(x_3quad, y_3quad, 'g^', markersize = 15)
    plt.plot(x_2quad, y_2quad, 'k*', markersize = 15)
    plt.plot(x_1quad, y_1quad, 'cp', markersize = 15)
    plt.grid()
    for i in [x_4quad, y_4quad, x_3quad, y_3quad, x_2quad, y_2quad, x_1quad, y_1quad]:
        print(i)
    
    # Expected output of the print statements:
    # [ 0.         -0.5        -0.70710678]
    # [ 0.          0.5        -0.70710678]
    # [ 0.70710678  0.8        -0.8         0.8        -0.8       ]
    # [ 0.70710678  0.8         0.8        -0.8        -0.8       ]
    # [ 0.8 -1.   1.5  0. ]
    # [ 1.   0.8 -0.5  1. ]
    # [ 1.  -1.   1.  -1.   1.5]
    # [ 1.   1.  -1.  -1.   1.5]
    
    ============
    Dependencies
    ============
    numpy (as np)
    '''
    # Centre all points by making the origin (x0, y0) 
    x = x - x0
    y = y - y0
    
    # Error check
    if return_type not in ['points','indices']:
        print(f'Warning: return_type not in ["points","indices"], setting it to "points"...')
        return_type = 'points'
    
    # Get indices of points
    idxs_4quad = x**2 + y**2 < R**2 # Points splitting the circle into 4 regions with non-zero area must lie in the circle.
    
    x_coords_at_least_R = np.abs(x) >= R
    y_coords_at_least_R = np.abs(y) >= R
    idxs_1quad = x_coords_at_least_R & y_coords_at_least_R # Points for whom the entire circle lies in a single quadrant must have (x,y) coordinates both exceeding (or equal) the circle radius
    
    idxs_3quad = ~idxs_4quad & ~x_coords_at_least_R & ~y_coords_at_least_R # Points splitting the circle into 3 regions with non-zero area must lie outside the circle, but not the square circumscribing the circle.
    idxs_2quad = ~( idxs_4quad | idxs_1quad | idxs_3quad ) # All other points split the circle into 2 regions with non-zero area. This is equivalent to ((~x_coords_at_least_R) ^ (~y_coords_at_least_R))
    
    if return_type == 'points':
        # Split points accordingly
        x_4quad, y_4quad = x[idxs_4quad], y[idxs_4quad]
        x_3quad, y_3quad = x[idxs_3quad], y[idxs_3quad]
        x_2quad, y_2quad = x[idxs_2quad], y[idxs_2quad]
        x_1quad, y_1quad = x[idxs_1quad], y[idxs_1quad]

        return x_4quad, y_4quad, x_3quad, y_3quad, x_2quad, y_2quad, x_1quad, y_1quad
    elif return_type == 'indices':
        return idxs_4quad, idxs_3quad, idxs_2quad, idxs_1quad
#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#
def TDKS_uniform_circle(x, y, x0 = 0, y0 = 0, R = 1, copy = False, normalise = True):
    '''
    For given vectors of points with x-coordinates in x and y-
    coordinates in y, compute the two-dimensional Kolmogorov-
    Smirnov statistic for equality of distribution to a uniform
    distribution over a circle centred at (x0, y0) with radius
    R.
    
    See: Kolmogorov-Smirnov Test for Two-Dimensional Data
         (Press, Teukolsky, 1988)
    
    ======
    Inputs
    ======
    x : numpy array of shape (n,)
        x-coordinates of input points
    y : numpy array of shape (n,)
        y-coordinates of input points
    x0 : float
        x-coordinate of circle centre
    y0 : float
        y-coordinate of circle centre
    R : float
        radius of circle
    copy : bool
        The corresponding argument to np.meshgrid
    normalise : bool
        If False, returns the un-normalised test statistic D
        (i.e., the maximum difference between observed and
        theoretical quadrant areas across all data points).
        If True, returns the normalised test statistic alpha*D,
        where alpha is a constant dependent on the sample size
        and the correlation coefficient between x and y.
        
    =======
    Outputs
    =======
    D : float
        The test statistic for the two-dimensional Kolmogorov-
        Smirnov test, normalised or un-normalised depending on 
        the value of normalise.
        
    ========
    Examples
    ========
    import numpy as np
    np.random.seed(2023)
    x, y = uniform_circle_points(n=100)
    print(TDKS_uniform_circle(x,y))
    
    # Expected output: 0.9380662357023231
    
    ============
    Dependencies
    ============
    numpy (as np), quadrant_fractions, quadrant_probabilities
    '''
    # Error check
    n = x.shape[0]
    if n != y.shape[0]:
        print(f'Warning: Number of x-coordinates ({n}) and y-coordinates ({y.shape[0]}) do not match!')
        
    # Compute differences in fractions between observed and theoretical quadrant areas
    top_right_observed, top_left_observed, bottom_left_observed, bottom_right_observed = quadrant_fractions(x, y, copy = copy)
    top_right_theoretical, top_left_theoretical, bottom_left_theoretical, bottom_right_theoretical = quadrant_probabilities(x, y, x0 = x0, y0 = y0, R = R)
    
    top_right_diff    = np.abs(top_right_theoretical-top_right_observed)
    top_left_diff     = np.abs(top_left_theoretical-top_left_observed)
    bottom_left_diff  = np.abs(bottom_left_theoretical-bottom_left_observed)
    bottom_right_diff = np.abs(bottom_right_theoretical-bottom_right_observed)
    
    # Return normalised or un-normalised test statistic depending on argument
    D = max(np.max(top_right_diff), np.max(top_left_diff), np.max(bottom_left_diff), np.max(bottom_right_diff))
    
    if normalise:
        r = np.corrcoef(x,y)[0,1]
        alpha = np.sqrt(n)/(1+np.sqrt(1-r**2)*(0.25-0.75/np.sqrt(n)))
        D = alpha*D
    
    return D
#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#
def uniform_circle_points(n = 100, R = 1, x0 = 0, y0 = 0, return_type = 'cartesian'):
    '''
    Generates points following a uniform distribution over a
    circle of radius R and centre in Cartesian coordinates
    (x0, y0).

    ======
    Inputs
    ======
    n : int
        Number of points to generate
    R : float
        Radius of circle
    x0 : float 
        x-coordinate of centre
    y0 : float
        y-coordinate of centre
    return_type : str in ['cartesian','polar']
        If 'cartesian', points are returned as x/y cartesian coordinates.
        If 'polar', points are returned as r/theta polar coordinates.
        
    =======
    Outputs
    =======
    A : numpy array of shape (n,)
        x-coordinates of input points if return_type == 'cartesian'
        r-coordinates of input points if return_type == 'polar'
    B : numpy array of shape (n,)
        y-coordinates of input points if return_type == 'cartesian'
        theta-coordinates of input points if return_type == 'polar'
    
    ========
    Examples
    ========
    import matplotlib.pyplot as plt
    import numpy as np
    x, y = uniform_circle_points(10000, return_type = 'cartesian')
    r, theta = uniform_circle_points(10000, return_type = 'polar')
    fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(4,8))
    ax[0].plot(x,y,'.',markersize=2) # Uniform distribution of points on the circle
    ax[0].grid()
    ax[1].plot(r,theta,'.',markersize=2) # Sparser on the left, denser on the right because the cicle has been stretched out
    ax[1].grid()
    plt.show()
    
    ============
    Dependencies
    ============
    numpy (as np), polar (from lcs_utils)
    '''
    if R < 0:
        print('Warning: Radius defined to be negative, taking the absolute value...')
        R = np.abs(R)
    if return_type not in ['cartesian','polar']:
        print('Warning: return_type not in ["cartesian","polar"], setting it to "cartesian"...')
        return_type = 'cartesian'
    
    r = R*np.sqrt(np.random.rand(n)) # Generate uniform distribution of distance from centre
    theta = np.random.rand(n)*2*np.pi # Generate uniform distribution of angle from centre
    x = x0 + r*np.cos(theta) # Get x-coordinates from r and theta
    y = y0 + r*np.sin(theta) # Get y-coordinates from r and theta
    
    if return_type == 'cartesian':
        return x, y
    elif return_type == 'polar':
        return polar(x, y)
#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#=========#
