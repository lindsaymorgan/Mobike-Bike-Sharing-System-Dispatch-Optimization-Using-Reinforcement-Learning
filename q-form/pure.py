import pandas as pd

need=pd.read_csv('2region_trip_20170510.csv')

initial_state=(2100,2100,0)

time_list=[6,10,15,21,24]
move_0_1=[30,1575,949,1688,238]
move_1_0=[63,1863,1578,2577,261]

state_0=(2100-move_0_1[0]+move_1_0[0],2100-move_1_0[0]+move_0_1[0],0)

