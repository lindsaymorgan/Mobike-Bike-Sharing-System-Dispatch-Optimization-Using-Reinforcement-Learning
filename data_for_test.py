import pandas as pd
import numpy as np
from collections import Counter

timelist=[i for i in range(25)]

date=20170510
data=pd.read_csv(f'data_{date}.csv')
data['starttime']=pd.to_datetime(data['starttime'])
data['startweidu'] = np.around(data['startweidu'], 13)
data['endjingdu'] = np.around(data['endjingdu'], 12)
data['startjingdu'] = np.around(data['startjingdu'], 12)
data['endweidu'] = np.around(data['endweidu'], 13)

data['startpos'] = list(zip(data['startweidu'], data['startjingdu']))
data['endpos'] = list(zip(data['endweidu'], data['endjingdu']))
point_set = set(data['startpos'])| set(data['endpos'])
point_list=sorted(list(point_set),key=lambda x: (x[0], x[1]))

#39.913941，116.397311
point_status = dict()
for i in point_list:
    if i[0]<39.913941:
        if i[1]<116.397311:
            point_status[i]=0
        else:
            point_status[i] = 1
    else:
        if i[1]<116.397311:
            point_status[i]=2
        else:
            point_status[i] = 3

data['startpos'] = [point_status[i] for i in data['startpos']]
data['endpos'] = [point_status[i] for i in data['endpos']]
data=data[data['startpos']!=data['endpos']]
data['trips']=[(i,j) for i,j in zip(data['startpos'],data['endpos']) ]
data['hour']=data['starttime'].dt.hour

tmp_pd=pd.DataFrame()
# tmp_pd.index=[(i,j) for i in range(4) for j in range(4)]
for i in range(len(timelist)-1):
    tmp=data[(data['hour']>=timelist[i]) & (data['hour']<timelist[i+1]) ]
    trip_dict=Counter(tmp['trips'])
    print(trip_dict)
    # print(pd.Series(trip_dict, name=f'{timelist[i+1]}'))
    tmp_pd=pd.merge(pd.DataFrame.from_dict(trip_dict, orient='index', columns=[f'{timelist[i ]}']), tmp_pd, left_index=True,
             right_index=True, how='outer')
    # tmp_pd=pd.merge(pd.Series(trip_dict, name=f'{timelist[i+1]}'), tmp_pd, left_index=True, right_index=True,how='outer')
    # tmp_pd=pd.concat([pd.Series(trip_dict, name=f'{timelist[i+1]}'),tmp_pd],axis=1)
    print(tmp_pd)

tmp_pd['start_region']=[i[0] for i in tmp_pd.index]
tmp_pd['end_region']=[i[1] for i in tmp_pd.index]

tmp_pd.fillna(0,inplace=True)
# tmp_pd=tmp_pd[['start_region','end_region','6','10','15','21','24']]
tmp_pd=tmp_pd[['start_region','end_region']+[f'{i}' for i in range(24)]]
tmp_pd.to_csv(f'4region_trip_{date}_eachhour.csv',index=0)