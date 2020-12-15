import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Setting global variables


figsize = (7.5, 6)
fontscale = 2.5
fontsize = 18
sns.set(font_scale = fontscale, style = 'whitegrid')
markersize = 75


labels_dict = {'lqd_assets': 'Liquid Assets ($)',
               'household_id': 'Households'
              }

ticks_dict = {'PESEX_x': ['Male', 'Female']
             }

def lqdassetsViz(df, targetdir, stat="count", weighted=False, both=False):
    """Visualizes the liquid assets of households in the Survey of Consumer Finances.
    
    Parameters
    
    Returns
    
    """
    if weighted & both:
        weights = df.weighting.values
        weights_str = " (Weighted and Unweighted)"
    elif weighted:
        weights = df.weighting.values
        weights_str = " (Weighted)"
    else:
        weights = None
        weights_str = " (Unweighted)"
        
    title = f"Cumulative Distribution of Liquid Assets\nAmong U.S. Households{weights_str}"
    xvar = 'lqd_assets'
    
    
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(15,12))
    sns.ecdfplot(x=df[xvar],
                 linewidth=4.5,
                 stat=stat,
                 weights=weights,
                 color='#8A2423'
                )
    if weighted & both:
        sns.ecdfplot(x=df[xvar],
                     linewidth=4.5,
                     stat=stat,
                     weights=None,
                     color='#303030'
                    )
        fig.legend(labels=["Weighted", "Unweighted"], loc="center right")
    
    # Adjust count for readability
    if stat == "count":
        ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # Labels and titles
    ax.set(title=f'{title}',
          xlabel=labels_dict[xvar]
          )
    
    #log scale
    ax.set(xscale="log")
    
    plt.tight_layout()
    
    fig.savefig(f'{targetdir}{title}.png', bbox_inches='tight')
    
    return  plt.show()

