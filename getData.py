import obspy
from obspy import UTCDateTime
from obspy.clients import fdsn

t1 = UTCDateTime(2023, 7, 20)
t2 = UTCDateTime(2023, 7, 22)
delta = 3600
stas = [2301, 2302, 2303, 2304, 2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2314, 2315, 2316]

stas = [2301]

DATASELECT = 'http://service.iris.edu/ph5ws/dataselect/1'
c = fdsn.client.Client(
    service_mappings={
        'dataselect': DATASELECT,
    },
)
c.set_credentials('t2kenyon@uwaterloo.ca', 'MtUqg4E6PI2ZQCDM')

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
            filename = 'data\{}_{}.mseed'.format(tr.id, tr.stats.starttime.strftime('%Y%m%d%H%M%S'))
            tr.write(filename)
    tNow = tNow + delta
