import obspy
from obspy import UTCDateTime
from obspy.clients import fdsn

t1 = UTCDateTime(2023, 7, 20) #Yr, Mo, Day. Can be expanded with commas to include Hr, Min, and Sec.
t2 = UTCDateTime(2023, 7, 22)
delta = 3600
stas = [2301, 2302, 2303, 2304, 2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2314, 2315, 2316] #total list of stations for ZE_2023 through ZE_2025#

stas = [2301] #Station to be tagged in the current data call#

DATASELECT = 'http://service.iris.edu/ph5ws/dataselect/1'  #Currently formatted to call PH5 data from the SAGE dataselect channel#
c = fdsn.client.Client(
    service_mappings={
        'dataselect': DATASELECT,
    },
)
c.set_credentials('', '')  #Note: this is only required to download restricted data sources from SAGE#

tNow = t1
req = []
while tNow < t2:
    print(tNow)
    for nowsta in stas:
        # req.append(('ZE', nowsta, '', 'GPZ', tNow, tNow+delta))
        print(nowsta)
        print('Getting data')
        S = c.get_waveforms('ZE', nowsta, '', 'GPZ', tNow, tNow+delta)
        print('Resampling')
        S.resample(250)
        print('Saving')
        for tr in S:
            filename = '[path needed]/{}_{}.mseed'.format(tr.id, tr.stats.starttime.strftime('%Y%m%d%H%M%S')) #saves to my own local directory, define something specific
            tr.write(filename)
    tNow = tNow + delta
