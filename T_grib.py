import pygrib
import numpy as np


def disp(fgrb, name=None, simple=True):
    """
    Display metadata information about a GRIB file, similar to MATLAB's ncdisp for NetCDF.
    
    Parameters:
    fgrb (str): Path to the GRIB file.
    """
    try:
        # Open the GRIB file
        with pygrib.open(fgrb) as grb:
            print(f"File: {fgrb}")
            print(f"Number of variables: {grb.messages}\n")

            # Iterate through each message and display metadata
            if simple is True:
                for i, message in enumerate(grb, start=1):
                    if name is not None and message.shortName != name:
                        continue
                    print(f"{message.shortName:<8}|{message}")
            else:
                for i, message in enumerate(grb, start=1):
                    if name is not None and message.shortName != name:
                        continue
                    print(f"{message.shortName}   {i}:")
                    print(f"    Name: {message.name}")
                    print(f"    Units: {message.units}")
                    print(f"    Date: {message.dataDate}")
                    print(f"    Time: {message.dataTime}")
                    print(f"    Forecast time: {message.forecastTime}")
                    print(f"    Level type: {message.typeOfLevel}")
                    print(f"    Level: {message.level}")
                    print(f"    Grid type: {message.gridType}")
                    print(f"    Latitude/Longitude box: {message.latitudeOfFirstGridPoint}, {message.longitudeOfFirstGridPoint} to {message.latitudeOfLastGridPoint}, {message.longitudeOfLastGridPoint}")
                    print("-" * 50)

    except Exception as e:
        print(f"Error reading the GRIB file: {e}")


def load_grid(fgrb):
    """
    Load latitude and longitude grids from a GRIB file.

    Parameters:
    fgrb (str): Path to the GRIB file.

    Returns:
    tuple: A tuple containing the longitude and latitude arrays.
    """
    try:
        # Open the GRIB file and load the coordinates
        with pygrib.open(fgrb) as grb:
            # Extract latitude and longitude grids from the first message
            lat, lon = grb.message(1).latlons()
            return lon, lat
    
    except Exception as e:
        print(f"Error loading grid from the GRIB file: {e}")
        return None


def load_data(fgrb, index):
    """
    Read data from a GRIB file by its message index.

    Parameters:
    fgrb (str): Path to the GRIB file.
    index (int): Index of the message to retrieve (starting from 1).

    Returns:
    tuple: A tuple containing:
        - data (numpy.ndarray): The data array of the selected message.
        - lons (numpy.ndarray): Longitude array.
        - lats (numpy.ndarray): Latitude array.
        - attributes (dict): Dictionary with metadata for the selected message.
    """
    try:
        with pygrib.open(fgrb) as grb:
            # Get the message by index
            message = grb.message(index)
            
            # Retrieve data and coordinates
            data = message.values
            lats, lons = message.latlons()
            
            # Extract metadata for this message
            attributes = {
                "parameter_name": message.name,
                "short_name": message.shortName,
                "units": message.units,
                "date": message.dataDate,
                "time": message.dataTime,
                "forecast_time": message.forecastTime,
                "level_type": message.typeOfLevel,
                "level": message.level,
                "grid_type": message.gridType,
            }

            # Print information about the variable
            print(f"Index {index:>3}: {message.shortName:<10}   {message.units:<10}   {message.typeOfLevel:<18}   {message.level:<8}")
            
            return data#, lons, lats, attributes

    except Exception as e:
        print(f"Error reading data at index {index} from the GRIB file: {e}")
        return None, None, None, None


