# Import package
import pandas as pd
import requests
import certifi
import io

# URL of the Excel file
url = "https://www.newyorkfed.org/medialibrary/media/research/capital_markets/allmonth.xls"

try:
    # Send a GET request to the URL with SSL verification enabled
    response = requests.get(url, verify=certifi.where())

    # Check if the request was successful
    if response.status_code == 200:
        # Attempt to read the Excel file (.xls format)
        try:
            # Use BytesIO to attempt to read the downloaded content as an Excel file
            data = pd.read_excel(io.BytesIO(response.content), engine='xlrd')
            print("Data imported successfully!")
            print(data.head())  # Display the first few rows of the dataframe
        except Exception as e:
            print("Failed to read the Excel file:", e)
    else:
        print("Failed to retrieve data. Status code:", response.status_code)
except Exception as e:
    print("An error occurred while downloading the file:", e)
