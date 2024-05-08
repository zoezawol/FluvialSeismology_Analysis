#%%
import waveformUtils
import obspy
S = obspy.read('PATH/ZE.2301*')

f, spectraF, spectraG, sttimes, tvec, RSAM = waveformUtils.plotSpectraTime(S, minfreq=1, maxfreq=100, winlength=7200,
                                                                           step=7200,
                                                                           spacing=2, ampScalar=3, normalize=True,
                                                                           specWin=60,
                                                                           specStep=30, rsamWin=1, saveFig=False)

f.show()

f1 = waveformUtils.multiDaySpectrogram(S, averageLength=3600, fftLength=60, minFreq=1.0, maxFreq=100, cmap='magma',
                                      dateLimits=None,
                                      vmin=0.4, vmax=0.9, plotAverage=True)

f1.show()
