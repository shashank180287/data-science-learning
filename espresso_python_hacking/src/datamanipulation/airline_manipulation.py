import os
from sklearn import preprocessing
import pandas as pd
from src.datamanipulation.data_preprocessing import make_col_positive, log_transform
import matplotlib.pyplot as plt
import logging

# TODO: Initialise a simple logger and set the desired format to be: TIME LEVEL-module-function-line number-message

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s:%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)

def transform_data(data):
    """
    Function to transform data according to some pre-defined steps.

    :param data: data to transform, dataframe format
    :return: transformed data
    """
    #logger.info(data.head())
    # TODO: drop column 'DAY_OF_WEEK'
    data = data.drop('DAY_OF_WEEK',1)
    #logger.info(data.head())
    # TODO: Rename column 'WHEELS_OFF' to 'HAS_WHEELS'
    data = data.rename(columns={'WHEELS_OFF': 'HAS_WHEELS'})
    #logger.info(data.HAS_WHEELS.to_string(index=False))
    # TODO: Fill blanks in column 'AIR_SYSTEM_DELAY' with the average of the values
    avrg = data["AIR_SYSTEM_DELAY"].mean()
    data["AIR_SYSTEM_DELAY"].fillna(avrg, inplace = True)
    #logger.info(data.AIR_SYSTEM_DELAY.to_string(index=False))
    # TODO: Scale values between 0 and 1 in 'DEPARTURE_DELAY' and put them in 'DEPARTURE_DELAY_NORMALISED'
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    data['DEPARTURE_DELAY_NORMALISED'] = scaler.fit_transform(data[["DEPARTURE_DELAY"]])
    #logger.info(data.DEPARTURE_DELAY_NORMALISED.to_string(index=False))
    # TODO: Make 'ARRIVAL_DELAY' column positive using a function imported from data_preprocessing.py
    data["ARRIVAL_DELAY"] = make_col_positive(data["ARRIVAL_DELAY"])
    #logger.info(data.head())
    # TODO: take the log of the column DEPARTURE_DELAY
    data["DEPARTURE_DELAY"] = log_transform(data["DEPARTURE_DELAY"])
    #logger.info(data.DEPARTURE_DELAY.to_string(index=False))
    return data


if __name__ == "__main__":
    '''
    HOMEWORK: Write a function that outputs insights into the data. I.e. aggregated data, plots etc. that will
    tell a compelling story to Heathrow about trends that we have discovered in the airline industry.
    
    The output should be the repository that helped produce the insight and a deck (.pdf, no longer that 5 slides)
    which would be used to present the insights to the client. 
    
    Please do not spend more than 3 hours on this.
    '''
    flights_data = pd.read_csv("../../data/flights_old.csv") # TODO: Import flight data
#    chunksize = 10 ** 6
#    for chunk in pd.read_csv("../../data/flights.csv", chunksize=chunksize):
    transformed = transform_data(flights_data)
    transformed.plot(x='DEPARTURE_DELAY_NORMALISED', y='ARRIVAL_DELAY', style='o')
    transformed.plot(x='ARRIVAL_TIME', y='ARRIVAL_DELAY', style='o')
    transformed.plot(x='DEPARTURE_TIME', y='ARRIVAL_DELAY', style='o')
    #plt.hist(transformed["DEPARTURE_DELAY_NORMALISED"], color="blue", bins=100)
    #plt.plot(transformed["DEPARTURE_TIME"], transformed["ARRIVAL_DELAY"])
    plt.show()

