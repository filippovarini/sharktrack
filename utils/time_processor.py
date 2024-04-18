from datetime import timedelta

def format_time(seconds):
    """
    Formats seconds to mm:ss:ms
    if 0 m, show 00 instead of 0
    ms should only be 2 digits
    """
    # Convert seconds to a timedelta
    td = timedelta(seconds=seconds)

    # Extract minutes, seconds, and milliseconds
    minutes = int(td.total_seconds() // 60)
    seconds = int(td.total_seconds() % 60)
    milliseconds = int(td.microseconds / 10000)  # Convert microseconds to milliseconds and round to 2 digits

    # Format the time string
    time_str = f"{minutes:02}m:{seconds:02}s"
    return time_str

def unformat_time(time_str):
    """
    Unformats time from mm:ss:ms to milliseconds
    """
    # Extract minutes, seconds, and milliseconds
    minutes, seconds, milliseconds = map(int, time_str.split(':'))
    milliseconds *= 10 # time_str has 2 digits for milliseconds, convert to 3 digits

    # Convert to milliseconds
    total_milliseconds = (minutes * 60 + seconds) * 1000 + milliseconds
    return total_milliseconds