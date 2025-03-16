import pandas as pd
import io
import csv
import config

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS

def detect_delimiter(file):
    """Detect the delimiter in a file by reading the first few lines."""
    sample = file.read(2048).decode("utf-8")  # Read the first 2 KB
    file.seek(0)  # Reset file pointer

    sniffer = csv.Sniffer()
    if sniffer.has_header(sample):
        return sniffer.sniff(sample).delimiter
    return ","  # Default to comma if delimiter detection fails

def valid_format(file):
    """Check if the uploaded file is a valid tab-delimited table."""
    try:
        # Read file as tab-separated values (TSV)
        df = pd.read_csv(io.StringIO(file.read().decode("utf-8")), sep="\t", header=0, low_memory=False)

        # Move file pointer back to the beginning (so Flask can save it later)
        file.seek(0)

        # If the file contains only one column, it's probably unstructured text
        if df.shape[1] == 1:
            return False, ["File does not appear to be a valid tab-separated table (only one column detected)."]

        # Get missing columns
        missing_columns = [col for col in config.REQUIRED_COLUMNS if col not in df.columns]

        if missing_columns:
            return False, missing_columns  # Return False and list of missing columns
        return True, None  # File is valid

    except pd.errors.ParserError as e:
        return False, [f"TSV Parsing Error: {str(e)}"]  # Handle tokenizing errors
    except Exception as e:
        return False, [str(e)]  # Catch other errors
