# Mark2_Rain_Backend_Interface
### Movtivation
Important step to quality control Mark2 rain model 
This code read csv files from backend. data files are generated from Mark2 data processing pipeline.
These dataframes are direct inputs for Mark2 rain model.


### Functionality
The script performs the following operations:

* Extracts data from CSV files in a specified folder.
* Processes the data to set the time column to datetime type and sort values.
* Merges three dataframes into one.
* Writes the merged dataframe to an AWS S3 bucket.
* Reads data from three separate CSV files in an AWS S3 bucket.
* Merges the data from the three files into one dataframe.
* Sorts the merged dataframe by time and sets the time column as the index.
* Removes rows with NaN values in columns fft1, fft2, and fft3.
* Drops duplicates in the resulting cleaned dataframe.

### Usage
* Save the code in a Python file with the name script.py.
* Place the CSV files to be processed in a folder and provide the path to that folder in the flpath variable at the beginning of the script.
* Replace the bucket variable in the script with the name of your own AWS S3 bucket.
* Run the script. It will generate a merged CSV file in the specified S3 bucket.
