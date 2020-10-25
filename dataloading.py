import requests
import zipfile
import pandas as pd
import numpy as np

from scipy import stats
from statsmodels.formula.api import ols

# Global variables

rename_dict = {'yy1': 'household_id',
                   'y1': 'imputed_hh_id',
                   'x42001': 'weighting',
                   'x7001': 'persons_in_PEU',
                   'x7020': 'spouse_part_of_PEU', ## 1 is not in PEU, 2 is in
                   'x102': 'ref_next_relative_type', ## 2 and 3 are spouses/partners, 1 is the respondent 
                   'x8000': 'switch_of_resp_ref',
                   'x14': 'ref_age',
                   'x19': 'spouse_age',
                   'x8021': 'ref_sex',
                   'x103': 'spouse_sex',
                   'x6809': 'ref_race',
                   'x6810': 'spouse_race',
                   'x5931': 'ref_educ', ## doctorate and profession combined as code 14
                   'x6111': 'spouse_educ',
                   'x6780': 'ref_UE_last_year', # UE or looking for work in last year
                   'x6784': 'spouse_UE_last_year',
                   'x7402': 'ref_industry_code', # 1 is agr, 2 energy/construction, 3 food, 4 trade/retail, 5 prof. services, 6 consumer services/goods, 7 public admin
                   'x7412': 'spouse_industry_code',
                   'x7401': 'ref_occ_code', # 1 is white-collar/services, 2 is front-line clerks/sales, 3 public institutions/entertainemnt front-line, 4 artisans, 5 industrial workers, 6 industrial managers
                   'x7411': 'spouse_occ_code',
                   'x501': 'primary_home_type', # 2 is mobile home, 2 is house/apt, 3 ranch, 5 farm
                   'x7136': 'chance_staying_home', # 0 - 100
                       'x6026': 'ref_mom_living',
                       'x6120': 'spouse_mom_living',
                       'x6032': 'ref_mom_educ',
                       'x6132': 'spouse_mom_educ',
                       'x6027': 'ref_mom_age', ## FOR THE PUBLIC DATA SET, PARENTS' AGES ROUNDED TO NEAREST AND TOP-CODED AT 95
                       'x6121': 'spouse_mom_age',
                       'x6028': 'ref_dad_living',
                       'x6122': 'spouse_dad_living',
                       'x6033': 'ref_dad_educ',
                       'x6133': 'spouse_dad_educ',
                       'x6029': 'ref_dad_age',
                       'x6123': 'spouse_dad_age',
                  'x5729': 'total_income',
                  'x7650': 'income_comparison', # 1 is high, 2 is low, 3 is normal compared to normal year
                  'x5802': 'inheritances',
                  'x6704': 'mutual_funds_value',
                  'x6706': 'bonds_mkt_value',
                  'x3915': 'stock_mkt_value',
                  'x6576': 'annuity_cash_value',
                  'x6587': 'trusts_cash_value',
                  'x4006': 'life_ins_cash_value',
                  'x414': 'total_cc_limit',
                  'x432': 'freq_cc_payment',
                  'x7575': 'rev_charge_accts'}

LOC_owed_list = ['x1108',
                     'x1119',
                     'x1130',
                     'x1136'
                    ]

educ_loans_owed_list = ['x7824',
                            'x7847',
                            'x7870',
                            'x7924',
                            'x7947',
                            'x7970',
                            'x7179'
                           ]

person_types_in_HH = ['x8020', 
                          'x102',
                          'x108',
                          'x114',
                          'x120',
                          'x126',
                          'x132',
                          'x202',
                          'x208',
                          'x214',
                          'x220',
                          'x226'
                         ]

cc_newcharges_list = ['x412',
                     'x420',
                     'x426'
                    ]

cc_currbal_list = ['x413',
                     'x421',
                     'x427'
                    ]

checking_accts_list = ['x3506',
                           'x3510',
                           'x3514',
                           'x3518',
                           'x3522',
                           'x3526',
                           'x3529'
                          ]

savings_accts_list = ['x3730',
                          'x3736',
                          'x3742',
                          'x3748',
                          'x3754',
                          'x3760',
                          'x3765'
                         ]
savings_accts_types_list = ['x3732',
                              'x3738',
                              'x3744',
                              'x3750',
                              'x3756',
                              'x3762'
                          ]


vars_for_calc = (LOC_owed_list
                + educ_loans_owed_list
                + cc_newcharges_list
                + cc_currbal_list
                + checking_accts_list
                + savings_accts_list
                + savings_accts_types_list)

sel_vars = (list(rename_dict.keys()) 
            + vars_for_calc
           )


def SCF_load_data(targetdir, year, series=None):
    """Loads SCF data for a given year into pandas data frame. Limited to 1989 and on. 
    
            Parameters:
                targetdir (str): String indicating where you want files saved.
                year (str or int): Indicating the year of SCF wanted.
                series (list of strings): Indicating subset of data requested.
            Returns:
                SCF_data (pd.df): Data frame of imported SCF data with labels adjusted 
                according to labels_dict in dataloading.py
    """
    # Set target zip file and relevant url
    targetzip = targetdir + f'SCF{str(year)}_data_public.zip'
    panel_string = 'p' if ((int(year)%3) != 0) else ''
    year = str(year)[-2:] if int(year) < 2002 else str(year)
    url = f'https://www.federalreserve.gov/econres/files/scf{str(year)}{panel_string}s.zip'
        
    # Return list of locations of extracted files   
    SCF_file_locs = URL_DL_ZIP(targetzip, targetdir, url) 
        
    # Read into pandas df    
    SCF_data = pd.read_stata(
        SCF_file_locs[0],
        #insert a list of variables or 'None' to get all
        columns=series)
    
    # Rename some variables of interest
    SCF_data.rename(columns=rename_dict, inplace=True)
    return SCF_data

def URL_DL_ZIP(targetzip, targetdir, url):
    """Downloads and unzips zip file from url and return locations of extracted filed.
    
            Parameters:
                targetzip (str): String indicating where zip file is to be saved.
                targetdir (str): String indicating where files are to be extracted.
                url (str): URL where the zip exists.
            Returns:
                file_locs (list of str): Returns locations for all the extracted files.
    """
        
    # Save Zip from archived site
    r = requests.get(url)
    with open(targetzip,'wb') as f: 
        f.write(r.content)
    
    # Unzipping file
    with zipfile.ZipFile(targetzip, 'r') as zip_ref:
        zip_ref.extractall(targetdir)
        # Get list of files names in zip
        files = zip_ref.namelist()
        
    # Return list of locations of extracted files   
    file_locs = [] 
    for file in files:
        file_locs.append(targetdir + file)
        
    return file_locs

def clean_df(df, query):
    
    
    

    ## Lines of credit

    df['LOC_owed_now'] = df[LOC_owed_list].sum(axis=1)

    ## Education loans

    df['ed_loans_owed_now'] = df[educ_loans_owed_list].sum(axis=1)




    ## CREDIT cards
    df['cc_newcharges_value'] = df[cc_newcharges_list].sum(axis=1)

    df['cc_currbal_value'] = df[cc_currbal_list].sum(axis=1)

    ## CHECKING Nos. 1-6 have detailed data, 7 is remaining accounts
    df['checking_accts_value'] = df[checking_accts_list].sum(axis=1)
    

    ## SAVINGS accts


    savings_accts_types = [df['x3732'],
                              df['x3738'],
                              df['x3744'],
                              df['x3750'],
                              df['x3756'],
                              df['x3762']
                          ]

    savings_accts_incl_codes = [1, 4, 12]

    # inlcuding only unincumbered savings
    for i, n in zip(savings_accts_list, savings_accts_types):
        df[i] = [(y if x in savings_accts_incl_codes 
                  else 0) 
                 for x, y in zip(n, df[i])]

    df['savings_accts_value'] = df[savings_accts_list].sum(axis=1)
    
    ## drop columns that are aggregated into columns calculated above
    df.drop(columns=vars_for_calc, inplace=True)
    
    ## liquid net worth 
    df['lqd_assets'] = (df['savings_accts_value']
                        + df['checking_accts_value']
                        + df['mutual_funds_value']
                        + df['bonds_mkt_value']
                        + df['stock_mkt_value']
                        + df['annuity_cash_value']
                        + df['trusts_cash_value']
                        + df['life_ins_cash_value']
                       )
    
    df['lqd_net_worth'] = df['lqd_assets'] - df['cc_currbal_value'] - df['rev_charge_accts']
    
    # education bins for reference person
    df['educ_bins'] = [(5 if x >= 14 
                        else (4 if x == 13
                             else (3 if x == 12
                                  else (2 if x >= 10
                                       else (1 if x >= 8
                                            else 0))))) for x in df['ref_educ']]
    for i, n in zip(range(5), ['doctorate_deg', 'professional_deg', 'master_deg', 'college_deg', 'hs_deg']):
        df[n] = [1 if x == (i+1) else 0 for x in df['educ_bins']]
        
    # education bins for reference person mom
    df['mom_educ_bins'] = [(2 if x == 12 
                        else (1 if x == 9
                             else 0)) for x in df['ref_mom_educ']]
    for i, n in zip(range(2), ['mom_college_deg', 'mom_hs_deg']):
        df[n] = [1 if x == (i+1) else 0 for x in df['mom_educ_bins']]
        
    # education bins for reference person dad
    df['dad_educ_bins'] = [(2 if x == 12 
                        else (1 if x == 9
                             else 0)) for x in df['ref_dad_educ']]
    for i, n in zip(range(2), ['dad_college_deg', 'dad_hs_deg']):
        df[n] = [1 if x == (i+1) else 0 for x in df['dad_educ_bins']]
    
    # weights/stats
    df['implicate'] = [x - y*10 for x, y in zip(df['imputed_hh_id'], df['household_id'])]
    ## weighting dividing by 5 since data implicates being combined for regression
    df['across_imp_weighting'] = [x/5 for x in df['weighting']]
    df['rel_weight'] = [ x/((df.weighting.sum())/5) for x in df['weighting']]
    
    #reducing to pop of interest
    df = df[query]
                        
    return df

def RII(df, Xseries, y):
    """Performs OLS linear regression for repeated-imputation inference predicting y with 
    Xseries and return p-values and coefficients.
    
            Parameters:
                df (pd.df): Pandas data frame containing features and target variable.
                Xseries (list of str): List containing features of model.
                y (str): Target variables of model.
            Returns:
                file_locs (list of str): Returns locations for all the extracted files.
    """
    list_coeff_vectors = {}
    list_var_vectors ={}
    coeff_var_vectors = {'coeff_vectors': list_coeff_vectors, 'var_covar': list_var_vectors}
    
    for i in range(5):
        # narrowing df to ith implicate
        df_imp = df[df.implicate == (i+1)] 
        
        # weighting each data point
        for n in list(df_imp.keys())[3:-3]:
            df_imp[n] = [float(x) * float(y) for x,y in zip(df_imp[n], df['across_imp_weighting'])]
            
        # regression of ith implicate dataset with constant
        formula = f'{y}~' + "+".join(Xseries) 
        lr = ols(formula=formula, 
             data=df_imp).fit()
        
        # vectors of regression coeff for ith implicate
        coeff_vector = np.array([lr.params.tolist()])
        list_coeff_vectors[i+1]=coeff_vector
        

        # var-covar matrix for ith impliciate
        var_covar_matrix = lr.cov_params().to_numpy()

        list_var_vectors[i+1]=var_covar_matrix
    
    output_dict = p_vals(coeff_var_vectors, Xseries=Xseries)
    
    return output_dict


def p_vals(output, Xseries):
    """Returns p-values given output of coefficients and variance-covariance matrices. 
    
            Parameters:
                output (dict of dicts):
                    'coeff_vectors': Dictionary with keys = implicate number and values 
                    being 1xk vectors of k coefficients in linear model for each implicate.
                    
                    'var_covar': Dictionary with keys = implicate number and values 
                    being kxk variance-covariance matrices for each implicate.
                    
                Xseries (list of str): Strings of independent variables in model.
            Returns:
                p_dict (dict of dicts): Keys are each of the independent variables and
                values are dictionaries containing p-values and coefficients.
    """
    
    #Instantiate coefficients from output
    coeffs = output['coeff_vectors']
    
    #num of imputations
    m = 5
    
    #num of coefficients
    k = len(coeffs[1][0])

    #  point estimates for each coeff 
    s = []
    for n in range(k):
        s.append([])


    for i in range(m):
        i += 1
        for n in range(k):
            s[n].append(coeffs[i][0][n])

    Qm_bar = []    
    for n in range(k):

        ssum = sum(s[n]) / m
        Qm_bar.append(ssum)

    Qm_bar = np.array([Qm_bar])    


    # var-cov matrix of point estimates
    summand_set = np.zeros((k, k))
    for i in range(m):
        i+=1
        var = coeffs[i] - Qm_bar

        summand = var.T * var

        summand_set = summand_set + summand

    Bm = summand_set / (m-1)


    # avg of variance-cov matrices
    summand_set = np.zeros((k,k))
    
    #instantiate var-covar matrices
    var_matrices = output['var_covar']
    
    
    for i in range(m):
        i+=1
        summand = var_matrices[i]

        summand_set = summand_set + summand

    Um_bar = summand_set/m


    # total variance of regression coeff.
    Tm = Um_bar + (1 + m**(-1))*Bm


    # std dev of regression coeff.
    Stddev = Tm**(1/2)

    # t stats of regression coeff.
    t_stats = Qm_bar/Stddev


    # Relative increase in variance due to nonresponse
    R_m =  ((1 + m**(-1))*Bm
          / Um_bar)

    # Degrees of freedom
    v = ((m-1)
         *(1+R_m**(-1))**(2))
    
    # P-values  
    p_values = []
    for i in range(k):
        p_values.append(stats.t.sf(abs(t_stats[i][i]), df=v[i][i])*2) 

    # Store p-values into dict    
    p_dict = {}
    X_vars = ['intercept'] + Xseries
    
    for i in range(k):
        p_dict[X_vars[i]] = {}
        p_dict[X_vars[i]]['p'] = p_values[i]
        p_dict[X_vars[i]]['coeff'] = Qm_bar[0][i]
   
    return p_dict  



"""EDUCATION
     1.    *1st, 2nd, 3rd, or 4th grade
                         2.    *5th or 6th grade
                         3.    *7th and 8th grade
                         4.    *9th grade
                         5.    *10th grade
                         6.    *11th grade
                         7.    *12th grade, no diploma
                         8.    *High school graduate - high school diploma or equivalent
                         9.    *Some college but no degree
                        10.    *Associate degree in college - occupation/vocation program
                        11.    *Associate degree in college - academic program
                        12.    *Bachelor's degree (for example: BA, AB, BS)
                        13.    *Master's degree ( for exmaple: MA, MS, MENG, MED, MSW, MBA)
                        14.    *Professional school degree (for example: MD, DDS, DVM, LLB, JD)
                        15.    *Doctorate degree (for example: PHD, EDD)
                        -1.    *Less than 1st grade
                         0.     Inap. (no spouse/partner;)
                    *********************************************************
                        FOR THE PUBLIC DATA SET, CODES 2, 3, 4, 5, 6, AND 7
                        ARE COMBINED WITH CODE 1; CODE 10 AND CODE 11 ARE
                        COMBINED WITH CODE 9, AND; CODES 13, 14, AND 15 ARE
                        COMBINED WITH CODE 12
                    *********************************************************

"""
    
"""SAVINGS accounts

                     1.    *TRADITIONAL SAVINGS ACCOUNT; "passbook account";
                            "statement account"
                     2.    *COVERDELL/EDUCATION IRA
                     3.    *529/STATE-SPONSORED EDUCATION ACCOUNT
                     4.    *MONEY MARKET ACCOUNT
                     5.     Christmas club account; other account for
                            designated saving purpose (e.g., vacation)
                     6.     Share account
                     7.    *HEALTH SAVINGS ACCOUNT; medical savings account
                    12.    *OTHER FLOATING-RATE SAVINGS ACCOUNT
                            (other than those coded 4)
                    14.     Informal group saving arrangement
                    20.     Foreign account type
                    30.    *SWEEP ACCOUNT n.e.c.; cash management account
                    -7.    *OTHER
                     0.     Inap. (no savings accounts: X3727^=1/fewer than 2
                            accounts: X3728<2/fewer than 3 account: X3728<3/
                            fewer than 4 accounts: X3728<4/fewer than 5
                            accounts: X3728<5/fewer than 6 accounts)
                *********************************************************
                    FOR THE PUBLIC DATA SET, CODES 6, 14, AND 20 ARE
                    COMBINED WITH CODE 1; CODES 3 AND 7 ARE COMBINED
                    WITH CODE 2; CODE 30 IS COMBINED WITH CODE 12
                *********************************************************
"""
    
"""Relatives in HH
                     1.    *RESPONDENT
                     2.    *SPOUSE; Spouse of R
                     3.    *PARTNER; Partner of R
                     4.    *CHILD (in-law) (of R or Spouse/Partner)
                     5.    *GRANDCHILD
                     6.    *PARENT
                     7.    *GRANDPARENT
                     8.    *AUNT/UNCLE
                     9.    *COUSIN
                    10.    *NIECE/NEPHEW
                    11.    *SISTER/BROTHER
                    12.    *GREAT GRANDCHILD
                    29.    *OTHER RELATIVE
                    31.    *ROOMMATE
                    32.    *FRIEND
                    34.    *BOARDER OR ROOMER/LODGER
                    35.    *PAID HELP; maid, etc.
                    36.    *FOSTER CHILD
                    39.    *OTHER UNRELATED PERSON
                     0.     Inap. (no further persons)
                *********************************************************
                    FOR THE PUBLIC DATA SET, CODE 12 IS COMBINED WITH
                    CODE 5; CODES 31, 32, AND 36 ARE COMBINED WITH CODE
                    39; CODES 9 AND 10 ARE COMBINED WITH CODE 29
                *********************************************************
"""