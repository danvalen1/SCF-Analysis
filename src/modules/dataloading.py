import requests
import zipfile
import pandas as pd

import waybackpy
from datetime import datetime
from dateutil.parser import parse
import os
import re

import numpy as np

from scipy import stats
from statsmodels.formula.api import ols

# Global variables

user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_0_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'

archive_age_limit = 30

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
                  'x7575': 'rev_charge_accts',
                  'x8300': 'num_fin_inst'
              }

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

negtozero_list = ['ref_race',
                  'ref_educ',
                  'spouse_educ',
                  'total_income',
                  'life_ins_cash_value'
                 ]


def SCF_load_data(targetdir, year, series=sel_vars):
    """Loads SCF data for a given year into pandas data frame. Limited to 1989 and on. 
    
            Parameters:
                targetdir (str): String indicating where you want files saved.
                year (str or int): Indicating the year of SCF wanted.
                series (list of strings): Indicating subset of data requested.
            Returns:
                SCF_data (pd.df): Data frame of imported SCF data with labels adjusted 
                according to labels_dict in dataloading.py
    """
    # Make targetdir if doesn't exist
    if not os.path.isdir(targetdir):
        os.mkdir(targetdir)
    
    # Set target zip file and relevant url
    panel_string = 'p' if ((int(year)%3) != 0) else ''
    year = str(year)[-2:] if int(year) < 2002 else str(year)
    url = f'https://www.federalreserve.gov/econres/files/scf{str(year)}{panel_string}s.zip'
    file = url.rsplit('/', 1)[-1]
    targetzip = targetdir + f'{file}'
       
    # Return list of locations of extracted files   
    SCF_file_locs = URL_DL_ZIP(targetzip, targetdir, url) 
    
    # account for potential case issues
    if series:
            series = [x.upper() for x in series]
        
    # Read into pandas df
    try:
        SCF_data = pd.read_stata(
            SCF_file_locs[0],
            #insert a list of variables or 'None' to get all
            columns=series)
    except:
        series = [x.lower() for x in series]
        SCF_data = pd.read_stata(
            SCF_file_locs[0],
            #insert a list of variables or 'None' to get all
            columns=series)
    
    # Rename some variables of interest
    SCF_data.columns = [x.lower() for x in SCF_data.columns]
    SCF_data.rename(columns=rename_dict, inplace=True)
    return SCF_data

def URL_DL_ZIP(targetzip, targetdir, url):
    """Downloads and unzips zip file from url and return locations of extracted files.
    
            Parameters:
                targetzip (str): String indicating where zip file is to be saved.
                targetdir (str): String indicating where files are to be extracted.
                url (str): URL where the zip exists.
            Returns:
                file_locs (list of str): Returns locations for all the extracted files.
    """
    archive_url = archive(url=url, targetdir=targetdir).get('archive_url')
    
    # Save Zip from archived site
    r_archive = requests.get(archive_url)
    r_url = requests.get(url)
    try:
        r = r_archive
        file_locs = Unzip(targetzip, r)
        
    except:
        r = r_url
        file_locs = Unzip(targetzip, r)
        
        
    return file_locs
    

def Unzip(targetzip, r):
    """Unzip a file from a url.
    
            Parameters:
                r (response object): Response from url where zip file exists.
                targetzip (str): String of path to target zip file.
            Returns:
                file_locs (list of str): Returns locations for all the extracted files.
    
    """
    
    with open(targetzip,'wb') as f: 
        f.write(r.content)
    
    # Set and/or create sub-folder
    sub_folder = targetzip.rsplit('.', 1)[0] +'/'
    try:
        os.mkdir(sub_folder)
    except:
        pass
    
    # Unzipping file        
    try:
        
        with zipfile.ZipFile(targetzip, 'r') as zip_ref:
            zip_ref.extractall(sub_folder)
            # Get list of files names in zip
            files = zip_ref.namelist()
    except:
        raise

        
        
    # Return list of locations of extracted files   
    file_locs = [] 
    for file in files:
        file_locs.append(sub_folder + file)
    
    return file_locs

def archive(url, targetdir=None):
    """Archives URL and saves information to data log in targetdir based on archive age limit.
    
            Parameters:
                url (str): String indicating where data exists.
                targetdir (str): String indicating where files' data log exists.
        
            Returns:
                archive_dict (dict): Dictionary with URL and timestamp of latest archive.
    """
    archive_dict = {'archive_url': None, 'archive_time': None}
    wayback_obj = waybackpy.Url(url=url, user_agent=user_agent)
    archive_age = len(wayback_obj.newest())
    print(archive_age)
    
    # Create new archive if age is greater than limit, else use most recent
    if archive_age > archive_age_limit:
        archive_dict['archive_url'] = wayback_obj.save().archive_url
        archive_dict['archive_time'] = datetime.utcnow()
        new_archive = 1
    else:
        archive_dict['archive_url'] = wayback_obj.newest().archive_url
        archive_dict['archive_time'] = wayback_obj.newest().timestamp
        new_archive = 0
    
    # Dict of data about archive
    d = {'URL': [url],
         'File': [url.rsplit('/', 1)[-1]],
         'Directory': [targetdir],
         'ArchiveURL': [archive_dict['archive_url']],
         'ArchiveTime': [archive_dict['archive_time']],
         'NewArchive': [new_archive]
        }
        
    # Add to or create data log
    try:
        data_log = pd.read_csv(f'{targetdir}+_data_log.csv', index_col='LogID')
        data_log['ArchiveTime'] = [parse(x) for x in data_log['ArchiveTime']]
        d['LogID'] = data_log.index.values.max() + 1
    
        d = pd.DataFrame.from_dict(d)
        d.set_index('LogID', inplace=True)
    
        data_log = pd.concat([d, data_log]).drop_duplicates(keep='last')
        
    except:
        d['LogID'] = 1
        data_log = pd.DataFrame(data=d)
        data_log.set_index('LogID', inplace=True)
   
    data_log.to_csv(f'{targetdir}+_data_log.csv', index_label='LogID')
    
    return archive_dict
    

def clean_SCF_df(df, neg_vals=False, query=None):
    # Averaging all values
    df = df.groupby("household_id").mean()

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
    savings_accts_incl_codes = [1, 4, 12]

    # inlcuding only unincumbered savings
    for i, n in zip(savings_accts_list, savings_accts_types_list):
        df[i] = [(y if x in savings_accts_incl_codes 
                  else 0) 
                 for x, y in zip(df[n], df[i])]

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
    
    # education bins for reference person
    df['educ_bins'] = [(5 if x >= 14 # Doctorate's and JDs/MDs
                        else (4 if x == 13 # Masters
                             else (3 if x == 12 # Bacehlors
                                  else (2 if x >= 10 # Associates
                                       else (1 if x >= 8 # HS
                                            else 0))))) for x in df['ref_educ']]
    for i, n in zip(range(5), ['doctorate_deg', 'master_deg', 'bachelor_deg', 'assoc_deg', 'hs_deg']):
        df[n] = [1 if x == (i+1) else 0 for x in df['educ_bins']]
        
    # Create target variable
    df['1k_target'] = [1 if x > 1000 else 0 for x in df['lqd_assets']]
    
    # Clean list of variables that have negative values
    if neg_vals == False:
        for var in negtozero_list:
            df[var] = [0 if x < 0 else x for x in df[var]]
    
    # fill nulls
    df.fillna(value=0, inplace=True)
    
    #reducing to pop of interest
    if query is not None:    
        df = df[query]
                        
    return df

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