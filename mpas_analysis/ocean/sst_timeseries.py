import os

from ..shared.plot.plotting import timeseries_analysis_plot

from ..shared.generalized_reader.generalized_reader \
    import open_multifile_dataset

from ..shared.timekeeping.utility import get_simulation_start_time, \
    date_to_days, days_to_datetime

from ..shared.analysis_task import setup_task

from ..shared.time_series import time_series

from ..shared.io.utility import build_config_full_path, make_directories


def sst_timeseries(config, streamMap=None, variableMap=None):
    """
    Performs analysis of the time-series output of sea-surface temperature
    (SST).

    config is an instance of MpasAnalysisConfigParser containing configuration
    options.

    If present, streamMap is a dictionary of MPAS-O stream names that map to
    their mpas_analysis counterparts.

    If present, variableMap is a dictionary of MPAS-O variable names that map
    to their mpas_analysis counterparts.

    Author: Xylar Asay-Davis, Milena Veneziani
    Last Modified: 04/08/2017
    """

    def compute_sst_part(timeIndices, firstCall):
        dsLocal = ds.isel(Time=timeIndices)
        return dsLocal

    print '  Load SST data...'

    # perform common setup for the task
    namelist, runStreams, historyStreams, calendar, namelistMap, streamMap, \
        variableMap, plotsDirectory = setup_task(config, componentName='ocean')

    simulationStartTime = get_simulation_start_time(runStreams)

    # get a list of timeSeriesStats output files from the streams file,
    # reading only those that are between the start and end dates
    startDate = config.get('timeSeries', 'startDate')
    endDate = config.get('timeSeries', 'endDate')
    streamName = historyStreams.find_stream(streamMap['timeSeriesStats'])
    fileNames = historyStreams.readpath(streamName, startDate=startDate,
                                        endDate=endDate,  calendar=calendar)
    print '\n  Reading files:\n' \
          '    {} through\n    {}'.format(
              os.path.basename(fileNames[0]),
              os.path.basename(fileNames[-1]))

    mainRunName = config.get('runs', 'mainRunName')
    preprocessedReferenceRunName = config.get('runs',
                                              'preprocessedReferenceRunName')
    preprocessedInputDirectory = config.get('oceanPreprocessedReference',
                                            'baseDirectory')

    movingAveragePoints = config.getint('timeSeriesSST', 'movingAveragePoints')

    regions = config.getExpression('regions', 'regions')
    plotTitles = config.getExpression('regions', 'plotTitles')
    regionIndicesToPlot = config.getExpression('timeSeriesSST',
                                               'regionIndicesToPlot')

    outputDirectory = build_config_full_path(config, 'output',
                                             'timeseriesSubdirectory')

    make_directories(outputDirectory)

    regionNames = config.getExpression('regions', 'regions')
    regionNames = [regionNames[index] for index in regionIndicesToPlot]

    # Load data:
    varList = ['avgSurfaceTemperature']
    ds = open_multifile_dataset(fileNames=fileNames,
                                calendar=calendar,
                                config=config,
                                simulationStartTime=simulationStartTime,
                                timeVariableName='Time',
                                variableList=varList,
                                variableMap=variableMap,
                                startDate=startDate,
                                endDate=endDate)

    ds = ds.isel(nOceanRegions=regionIndicesToPlot)

    yearStart = days_to_datetime(ds.Time.min(), calendar=calendar).year
    yearEnd = days_to_datetime(ds.Time.max(), calendar=calendar).year
    timeStart = date_to_days(year=yearStart, month=1, day=1,
                             calendar=calendar)
    timeEnd = date_to_days(year=yearEnd, month=12, day=31,
                           calendar=calendar)

    if preprocessedReferenceRunName != 'None':
        print '  Load in SST for a preprocesses reference run...'
        inFilesPreprocessed = '{}/SST.{}.year*.nc'.format(
            preprocessedInputDirectory, preprocessedReferenceRunName)
        dsPreprocessed = open_multifile_dataset(
            fileNames=inFilesPreprocessed,
            calendar=calendar,
            config=config,
            simulationStartTime=simulationStartTime,
            timeVariableName='xtime')
        yearEndPreprocessed = days_to_datetime(dsPreprocessed.Time.max(),
                                               calendar=calendar).year
        if yearStart <= yearEndPreprocessed:
            dsPreprocessedTimeSlice = \
                dsPreprocessed.sel(Time=slice(timeStart, timeEnd))
        else:
            print '   Warning: Preprocessed time series ends before the ' \
                'timeSeries startYear and will not be plotted.'
            preprocessedReferenceRunName = 'None'

    cacheFileName = '{}/sstTimeSeries.nc'.format(outputDirectory)

    dsSST = time_series.cache_time_series(ds.Time.values, compute_sst_part,
                                          cacheFileName, calendar,
                                          yearsPerCacheUpdate=10,
                                          printProgress=True)

    print '  Make plots...'
    for index, regionIndex in enumerate(regionIndicesToPlot):

        title = plotTitles[regionIndex]
        title = 'SST, %s, %s (r-)' % (title, mainRunName)
        xLabel = 'Time [years]'
        yLabel = '[$^\circ$ C]'

        SST = dsSST.avgSurfaceTemperature.isel(nOceanRegions=index)


        figureName = '{}/sst_{}_{}.png'.format(plotsDirectory,
                                               regions[regionIndex],
                                               mainRunName)

        if preprocessedReferenceRunName != 'None':
            SST_v0 = dsPreprocessedTimeSlice.SST

            title = '{}\n {} (b-)'.format(title, preprocessedReferenceRunName)
            timeseries_analysis_plot(config, [SST, SST_v0],
                                     movingAveragePoints,
                                     title, xLabel, yLabel, figureName,
                                     lineStyles=['r-', 'b-'],
                                     lineWidths=[1.2, 1.2],
                                     calendar=calendar)
        else:
            timeseries_analysis_plot(config, [SST], movingAveragePoints, title,
                                     xLabel, yLabel, figureName,
                                     lineStyles=['r-'], lineWidths=[1.2],
                                     calendar=calendar)
