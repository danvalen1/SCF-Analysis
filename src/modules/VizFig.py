import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Setting global variables

dpi = 300
figsize = (7.5, 6)
fontscale = 2
fontsize = 18
sns.set(font_scale = fontscale, style = 'whitegrid')
markersize = 75


labels_dict = {'Features': 'Features',
               'LogRegCoeff': 'Logistic Regression\n Coefficients'
               
              }

ticks_dict = {'LogRegCoeff': ['Num. of Fin. Inst.', 
                              'Total Cred. Card Limit', 
                              'Cred. Card New Charges', 
                              'Bachelor Degree', 
                              'White-Collar Occupation'
                             ]

             }







def PlotCatCoeff(df, xvar,targetdir, yvar=None, orient=None, kind='count',palette="coolwarm_r"):
    """Plot a categorical plot with automatic labels provided in a global dict in CustomModule. 
        Pass a dataframe through `df`, a string through `xvar`, and where to save the image through
        `targetdir`. 
    """
    
    title = f'Scale of {labels_dict[xvar]}'
    

    fig = sns.catplot(data=df,
                x=xvar,
                y=yvar,
                kind = kind,
                orient = orient,
                height = figsize[1],
                aspect = figsize[0]/figsize[1],
                palette=palette
               )
    plt.ylabel(yvar)
    plt.xlabel('Scale')
    plt.title(title)
    labels = ticks_dict[xvar]
    plt.yticks(list(range(len(labels))), labels)

    fig.savefig(f'{targetdir}{title}.png', bbox_inches='tight')

    return plt.show()
