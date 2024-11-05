import os
import requests
from datetime import datetime, timedelta
import netCDF4 as nc
import cdsapi


def download(url, filename):
    """Helper function to download a file from a URL."""
    try:
        print(f"Downloading {url} ...")

        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Saved: {filename}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False
    return True


def download_ERA5(start_time, end_time, interval_time=1, outdir="."):
    """
    Downloads ERA5 0.25-degree reanalysis data for the specified time range.
    Parameters:
        start_time (str): Start time in 'YYYY-MM-DD_HH' format.
        end_time (str): End time in 'YYYY-MM-DD_HH' format.
        interval_time (int): Time interval in hours (default is 1).
        outdir (str): Directory to save the downloaded files.
    Example:
        download_ERA5_0p25('2022-12-10_00', '2022-12-10_06', 1, './data')
    """
    # Initialize the CDS API client
    client = cdsapi.Client(quiet=True, debug=False)
    # Convert input times to datetime objects
    start = datetime.strptime(start_time, "%Y-%m-%d_%H")
    end = datetime.strptime(end_time, "%Y-%m-%d_%H")
    # Ensure the output directory exists
    os.makedirs(outdir, exist_ok=True)
    # Iterate over the time range in the given interval
    current_time = start
    while current_time <= end:
        # Generate the date and time strings for the request
        year = str(current_time.year)
        month = f"{current_time.month:02d}"
        day = f"{current_time.day:02d}"
        hour = f"{current_time.hour:02d}:00"
        # Create the request payload
        request = {
            "product_type": "reanalysis",
            "variable": [
                "2m_dewpoint_temperature",
                "2m_temperature",
                "mean_sea_level_pressure",
                "sea_surface_temperature",
                "surface_pressure",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "surface_latent_heat_flux",
                "surface_sensible_heat_flux",
                "surface_net_solar_radiation",
                "surface_net_thermal_radiation",
                "total_cloud_cover",
                "evaporation",
                "total_precipitation"
            ],
            "year": year,
            "month": month,
            "day": day,
            "time": hour,
            "format": "netcdf",  # Corrected key
            "area": [41, 117, 34, 127],  # North, West, South, East
        }
        # Generate the output filename
        filename = f"ERA5_{year}{month}{day}_{hour.replace(':', '')}.nc"
        filepath = os.path.join(outdir, filename)
        print(f"Downloading: {filename}...")
        # Retrieve and download the data
        client.retrieve("reanalysis-era5-single-levels", request).download(filepath)
        print(f"Saved to: {filepath}")
        # Increment the time by the specified interval
        current_time += timedelta(hours=interval_time)
    print("---------------Download complete.---------------")


def load_grid_ERA5(fera5):
    ds = nc.Dataset(fera5, mode='r')
    # Read the 'longitude' and 'latitude'
    x = ds.variables['longitude'][:]
    y = ds.variables['latitude'][:]
    # Close the dataset
    ds.close()

    GRID = {'x': x, 'y': y}

    return GRID
    



