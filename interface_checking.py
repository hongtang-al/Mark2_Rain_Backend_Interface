import pandas as pd
import numpy as np
import os
from utils import df_to_s3, df_from_s3


flpath="./data/7_28/"


def extract_dfs(flpath):
    '''  Extracts data from CSV files in the specified folder and returns three pandas dataframes.
        read files pulled from Fei including enriched rh
    Args:
        flpath (str): A string representing the path of the folder containing the CSV files.
    Returns:
        Tuple: A tuple of three pandas dataframes representing the extracted data.
    Raises:
        FileNotFoundError: If the specified folder path does not exist or is invalid.
    '''

    folder_path = flpath
    
    if not os.path.exists(flpath):
        raise FileNotFoundError(f"The specified folder path '{flpath}' does not exist or is invalid.")

    dfcalibrated=dfdsd = dfinternal= pd.DataFrame()

    for filename in os.listdir(folder_path):
        
        if filename.endswith(".csv"):
            df_name = filename.split(".")[0]

            if ('calibrated'  in df_name): #type(df_name ).__name__
                _cal = pd.read_csv(os.path.join(folder_path, filename))
                dfcalibrated = pd.concat([dfcalibrated,_cal])
                # print(_cal.shape)
            elif  'dsd_raw'  in df_name:
                _dsd = pd.read_csv(os.path.join(folder_path, filename))
                dfdsd = pd.concat([dfdsd,_dsd])
            elif  'internal' in df_name:
                _int = pd.read_csv(os.path.join(folder_path, filename))
                dfinternal = pd.concat([dfinternal,_int])
                
    return dfcalibrated, dfdsd, dfinternal


dfcalibrated, dfdsd, dfinternal=extract_dfs( flpath )

# set time for three dataframes to be datatime type for merge files
dfcalibrated['time']=pd.to_datetime(dfcalibrated['time'])
dfdsd['time']=pd.to_datetime(dfdsd['time'])
dfinternal['time']=pd.to_datetime(dfinternal['time'])

#sort values
dfcalibrated = dfcalibrated.sort_values(by='time')
dfdsd = dfdsd.sort_values(by='time')
dfinternal = dfinternal.sort_values(by='time')

# merge three dataframes into one
merged_df = pd.merge_asof(dfcalibrated, dfdsd, on=['time'], tolerance=pd.Timedelta('5 minutes'))

merged_df = pd.merge_asof(merged_df, dfinternal, on=['time'], tolerance=pd.Timedelta('5 minutes'))

bucket='arable-adse-dev'
key='Carbon Project/Stress Index/UCD_Almond/merged_rain_check2.csv'

df_to_s3(merged_df,  key, bucket)


# ### merge all data 

merge_data0 = df_from_s3('Carbon Project/Stress Index/UCD_Almond/merged_rain_check.csv', 'arable-adse-dev')
merge_data1 = df_from_s3('Carbon Project/Stress Index/UCD_Almond/merged_rain_check1.csv', 'arable-adse-dev')
merge_data2 = df_from_s3('Carbon Project/Stress Index/UCD_Almond/merged_rain_check2.csv', 'arable-adse-dev')

merge_data_all=pd.concat([merge_data0,merge_data1,merge_data2])

#write merged dataframe to s3

df_to_s3(merge_data_all,  'Carbon Project/Stress Index/UCD_Almond/merged_rain_check.csv', bucket)

merge_data_all = merge_data_all.sort_values(by='time')
merge_data_all.set_index('time', inplace =True)


#remove data with NaNs
merge_data_all_clean = merge_data_all.dropna(subset=['fft1', 'fft2', 'fft3'], how='all')

merge_data_all_clean=merge_data_all_clean.drop_duplicates()
merge_data_all_clean.head()


