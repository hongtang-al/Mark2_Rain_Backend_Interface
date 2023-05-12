import pandas as pd
import io
import boto3
import json
import pandas.io.sql as psql
import psycopg2 as pg
import sys
import numpy as np
import geopandas as gpd
sys.path.append('/home/ec2-user/SageMaker/adse/lib')

from helper import *
from sqlalchemy import create_engine, text
from utils import df_from_s3, df_to_s3

bucket_name = 'arable-adse-dev'
path = f'rain_classification_april_2022/rain_correlation_analysis.csv'

pull_data=False
if pull_data:
    def get_user_db_creds(user: str, environment: str):
        client = boto3.client("secretsmanager")
        response = client.get_secret_value(SecretId=f"{user}_db_creds_1")
        secret_db_creds = json.loads(response["SecretString"])
        db_info = {
            "user": secret_db_creds[f"user_{environment}"],
            "password": secret_db_creds[f"password_{environment}"],
            "host": secret_db_creds[f"host_{environment}"],
            "db": secret_db_creds[f"db_{environment}"],
            "port": secret_db_creds[f"port_{environment}"],
        }
        return db_info

    def connect_db(dsn: str) -> str:
        cnx = create_engine(dsn)
        return cnx

    def read_daily(cnx):
    
    schema_raw = 'daily'
    query_template_raw = """   
    --may want to change me here
    with
    failure_sensors as (
    select
    date_trunc('hour', lh.time) as date
    , lh.device
    , lh."location"
    , count(lh.time) as raw_sync
    ,sum(case when p = -1000 then 1 else 0 end) as p_ct,
        sum(case when rh = -1000 then 1 else 0 end) as rh_ct, 
        sum(case when temp = -1000 then 1 else 0 end) as temp_ct,
        sum(case when therm_temp = -1000 then 1 else 0 end) as therm_temp_ct,
        sum(case when pres_temp = -1000 then 1 else 0 end) as press_temp_ct,
        sum(case when press_3 = -1000 then 1 else 0 end) as p3_ct
    from device_data.raw lh  
    where lh.time between '2022-10-01' and '2023-03-01'
    and lh.device in ('C003015', 'C003017', 'C003188', 'C003231', 'C003240', 'C003276',
        'C003279', 'C003398', 'C003655', 'C004146', 'C004149', 'C004183',
        'C004196', 'C004223', 'C004231', 'C004299', 'C004301', 'C004830',
        'C004854', 'C004894', 'C004976', 'C004988', 'C005042', 'C005043',
        'C005182', 'C005285', 'C005312', 'C005348', 'C005386', 'C005429',
        'C006158', 'C006160', 'C006164', 'C006826', 'C007852', 'C007880',
        'C007978')
    group by 1,2,3
    )
    select
    date_trunc('hour',h.time) as date
    , h.device
    , h."location"
    , l.name as location_name
    , l.country
    , fw
    , hr.precip
    , hr.wind_speed_max
    , hr.precip_version
    , hr.precip_classifier_version
    , hr.tair
    , hr.rh
    , hr.p
    , hr.swdw
    , raw_sync
    , p_ct
    , rh_ct
    , temp_ct
    , therm_temp_ct
    , p3_ct
    , press_temp_ct
    , avg(batt_current) as avg_batt_current
    , avg(batt_pct) as avg_batt_pct
    , min(batt_volt) as min_batt_volt
    , COUNT("reset") filter (where reset=0) as reset_0
    , COUNT("reset") filter (where reset=1) as reset_1
    , COUNT("reset") filter (where reset=2) as reset_2
    , COUNT("reset") filter (where reset=5) as reset_5
    , COUNT("reset") filter (where reset=6) as reset_6
    , COUNT("reset") filter (where reset=8) as reset_8
    , COUNT("reset") filter (where reset=9) as reset_9
    , COUNT("reset") filter (where reset=12) as reset_12
    , COUNT("reset") filter (where reset=13) as reset_13
    , COUNT("reset") filter (where reset=14) as reset_14
    from device_data.health h
    join model_data."location" l on h."location" = l.id
    left join device_data.hourly hr on h.device = hr.device and date_trunc('hour',h.time) = hr.time
    left join failure_sensors as sf on h."location" = sf.location and date_trunc('hour',h.time) = sf.date
    where h.device in ('C003015', 'C003017', 'C003188', 'C003231', 'C003240', 'C003276',
        'C003279', 'C003398', 'C003655', 'C004146', 'C004149', 'C004183',
        'C004196', 'C004223', 'C004231', 'C004299', 'C004301', 'C004830',
        'C004854', 'C004894', 'C004976', 'C004988', 'C005042', 'C005043',
        'C005182', 'C005285', 'C005312', 'C005348', 'C005386', 'C005429',
        'C006158', 'C006160', 'C006164', 'C006826', 'C007852', 'C007880',
        'C007978')
    and h.time between '2022-10-01' and '2023-03-01'
    group by 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21
    """


    sql_query = query_template_raw.format(schema=schema_raw)


    df = pd.read_sql_query(sql_query, cnx)


    return df

    # retrieve personal tocken from arable secrete Manager
    # --may want to change me here
    dsn=get_user_db_creds('hong_tang', 'alp')
    sqlalchemy_dsn = 'postgresql://{user}:{password}@{host}:{port}/{db}'.format(**dsn)
    pg_conn = connect_db(sqlalchemy_dsn)

    df = read_daily(pg_conn, device, column_hourly, start_date, end_date)
    df['date'] = pd.to_datetime(df['date'])

    df_to_s3(df, path, bucket_name, format ='csv')

df = df_from_s3(path, bucket_name, format ='csv')
df["date"] = pd.to_datetime(df["date"])

calval_daily=df_from_s3('mark3_tbelow/joined_calval_precip_ref_dsd_hourly.csv', bucket='arable-adse-dev')
calval_daily["time"] = pd.to_datetime(calval_daily["time"])
calval_daily = calval_daily.rename(columns={'site_id':'location_name_s','time':'date'})
calval_daily = calval_daily.drop(['precip'],axis=1)

inner = df.merge(calval_daily, on=['date','device'], how='inner')
inner = inner[inner["ref_precip"]<1000]
inner["under"] = np.where((inner["precip"] < (inner["ref_precip"])), 1, 0)
inner["over"] = np.where((inner["precip"] > (inner["ref_precip"])), 1, 0)
inner["device_type"] = inner["device"].apply(lambda x: x[0:4])
inner["diff_precip"] = np.abs((inner["precip"]) - (inner["ref_precip"]))

notincludelist=['C004223', 'C004299', 'C003655', 'C005042', 'C005043', 'C005429','C004149']
inner = inner.loc[~inner.device.isin(notincludelist) ]

inner_under = inner[inner["under"] == 1][['date', 'device', 'location', 'location_name', 'country', 'fw','precip', 
                                          'wind_speed_max', 'precip_version','precip_classifier_version', 'tair', 
                                          'rh', 'p', 'swdw','raw_sync', 'p_ct','rh_ct', 'temp_ct', 'p3_ct', 'press_temp_ct','therm_temp_ct',
                                          'avg_batt_current', 'avg_batt_pct','min_batt_volt', 'reset_0', 'reset_1', 'reset_2', 'reset_5', 
                                          'reset_6','reset_8', 'reset_9', 'reset_12', 'reset_13', 'reset_14', 
                                          'ref_precip', 'avg_dsd_sum','device_type']].reset_index(drop=True)
inner_under_noprec = inner_under[inner_under["precip"] == 0][['date', 'device', 'location', 'location_name', 'country', 'fw','precip', 
                                          'wind_speed_max', 'precip_version','precip_classifier_version', 'tair', 
                                          'rh', 'p', 'swdw','raw_sync', 'p_ct','rh_ct', 'temp_ct', 'p3_ct', 'press_temp_ct','therm_temp_ct',
                                          'avg_batt_current', 'avg_batt_pct','min_batt_volt', 'reset_0', 'reset_1', 'reset_2', 'reset_5', 
                                          'reset_6','reset_8', 'reset_9', 'reset_12', 'reset_13', 'reset_14', 
                                          'ref_precip', 'avg_dsd_sum','device_type']].reset_index(drop=True)

display(inner_under)
display(inner_under_noprec)

inner_under_noprec.isnull().sum()

inner_under_noprec[(inner_under_noprec["p_ct"] >0)]

inner_under_noprec[(inner_under_noprec["rh_ct"] >0)]

inner_under_noprec[(inner_under_noprec["therm_temp_ct"] >0)]

inner_under_noprec["precip_version"].value_counts()/len(inner_under_noprec)

inner_under_noprec["precip_version"].value_counts()
