#Merge all features into a single csv file
import pandas as pd
import numpy as np

#probably a better way to do this. We make binary columns from labeled string features.
def convertToBinaryColumns(inspections, inputDF, column, targetString):
    #we sort the values of Types, and move nan to the end, for consistency with our sum features later on.
    Types = inputDF[column].unique()
    Types = sorted([x for x in Types if str(x) != 'nan'])
    Types = np.array(Types,dtype=object)
    Types = np.append(Types, [np.nan])
    inputDF[column] = inputDF[column].map(dict(zip(Types, range(len(Types)))))

    gb = inputDF.groupby(['camis', 'inspection_date'])
    n = len(inputDF[column].unique())
    
    binaryFeatures = [[0]*len(inspections['camis']) for _ in range(n)]
    
    for i in range(len(inspections['camis'])):
        for v in gb.get_group((inspections['camis'].values[i],inspections['inspection_date'].values[i]))[column].values:
            binaryFeatures[v][i] = 1

    for v in range(n):
        inspections[targetString + str(v)] = binaryFeatures[v]

def numberPassesPrior(inspections, gb):
    passfail = [[0]*len(inspections['camis']) for _ in range(2)]
    previous = [0]*len(inspections['camis'])
    
    for i in range(len(inspections['camis'])):
        if inspections['camis'].values[i] in gb.groups.keys():
            camisResults = gb.get_group(inspections['camis'].values[i])
            for v in range(len(camisResults['passed'])):
                if camisResults['inspection_date'].values[v] < inspections['inspection_date'].values[i]:
                    if camisResults['passed'].values[v] == 1:
                        passfail[0][i] += 1
                        previous[i] = 1
                    else:
                        passfail[1][i] += 1
                        previous[i] = -1
                else:
                    #we take no information from the future. We only look at how many passes / fails occured PRIOR to the inspection.
                    break

    inspections['PassedPrior'] = passfail[0]
    inspections['FailedPrior'] = passfail[1]
    inspections['priorResult'] = previous

def computeScoreFromGuess(inspections, guess, targetString):
        scores = [0]*len(inspections['camis'])
        v_gb = violations.groupby(['camis', 'inspection_date'])
        for i in range(len(inspections['camis'])):
            for v in v_gb.get_group((inspections['camis'].values[i],inspections['inspection_date'].values[i]))['violation_description'].values:
                scores[i] += guess[v]
        inspections[targetString] = scores

#inspections = pd.read_csv('inspections_train.csv', parse_dates=['inspection_date'])
inspections = pd.read_csv('inspections_test.csv', parse_dates=['inspection_date']) #uncomment to create kaggle data
violations = pd.read_csv('violations.csv', parse_dates=['inspection_date'])
venue_stats = pd.read_csv('venues.csv').set_index('camis')

#add number of occurances of resturant in venue dataset as a feature.
#Idea is that chain resturants may on average perform differently
#than unique establishments
venue_stats = venue_stats.merge(venue_stats['dba'].value_counts(), 'left', left_on = 'dba', right_index = True, suffixes = ('', '_counts'))

#map each returant type to an integer
cuisines = venue_stats['cuisine_description'].unique()
cuisines.sort()
venue_stats['cuisine_description'] = venue_stats['cuisine_description'].map(dict(zip(cuisines, range(len(cuisines)))))

boro = venue_stats['boro'].unique()
boro.sort()
venue_stats['boro'] = venue_stats['boro'].map(dict(zip(boro, range(len(boro)))))

street = venue_stats['street'].unique()
street = sorted([x for x in street if str(x) != 'nan'])
street = np.array(street,dtype=object)
street = np.append(street, [np.nan])
venue_stats['street'] = venue_stats['street'].map(dict(zip(street, range(len(street)))))

#more merges involving zip code data
# (waiting on Paul to gather data for here)
#
#zipCodeData = pd.read_csv('zipLatLong.csv').set_index('zipcode')
#zipCodeData = zipCodeData.drop(columns = ['Unnamed: 0'])
#venue_stats = venue_stats.merge(zipCodeData, 'left', left_on='zipcode', right_index=True)

venue_stats = venue_stats.drop(columns = ['dba'])
inspections = inspections.merge(venue_stats, 'left', left_on='camis', right_index=True)

#Bring in data from the NOAA on percipitation, maximal and minimal temperature in Central Park
weather = pd.read_csv(r'C:\Users\Eugene\Documents\Kaggle\weather.csv',parse_dates=['inspection_date']).set_index('inspection_date')
w_gb = weather.groupby('NAME')
weatherStats = w_gb.get_group('NY CITY CENTRAL PARK, NY US')
weatherStats.drop(columns = ['STATION','TAVG','TOBS'], inplace = True)
weatherStats['TAVG'] = (weatherStats['TMAX'].values + weatherStats['TMIN'].values)/2
inspections = inspections.merge(weatherStats, 'left', left_on='inspection_date', right_index=True)

#now we move onto violation information, here we are most interested in binary conversion
convertToBinaryColumns(inspections, violations,'violation_description','vBin')
convertToBinaryColumns(inspections, violations,'inspection_type','iBin')
convertToBinaryColumns(inspections, violations,'action','aBin')

#here we will inclue all of the sum features we constructed
# (for now we exclude the sum which was optimized with month values)
GuessDistribution = pd.read_csv('First_Guess_Values.csv')['guesses'].values/13
#D1 = pd.read_csv('Guess_Values.csv')['guesses'].values
#D2 = pd.read_csv('Guess_Values_2.csv')['guesses'].values
computeScoreFromGuess(inspections, GuessDistribution, 'SimpleGuess')
#computeScoreFromGuess(inspections, D1, 'optimizedGuess')
#computeScoreFromGuess(inspections, D2, 'optimizedConstrainedGuess')

D1 = pd.read_csv('Simple_Guess_Final.csv')['guesses'].values
D2 = pd.read_csv('Simple_Guess_Final_Rounded.csv')['guesses'].values
D3 = pd.read_csv('MatlabFinalSum1.csv')['guesses'].values
D4 = pd.read_csv('MatlabFinalSum2.csv')['guesses'].values

computeScoreFromGuess(inspections, D1, 's1')
computeScoreFromGuess(inspections, D2, 's2')
computeScoreFromGuess(inspections, D3, 's3')
computeScoreFromGuess(inspections, D4, 's4')

#Introduce new feature: number of passed/failed inspections prior to this inspection
#note that we read in inspections again because when running this file for test data,
#we will not have access to inspections passed column.
PassFail = pd.read_csv('inspections_train.csv', parse_dates=['inspection_date'])
pf_gb = PassFail[['camis','inspection_date','passed']].groupby(['camis'])
numberPassesPrior(inspections, pf_gb)

inspections.to_csv('All_Features.csv',index=False)
