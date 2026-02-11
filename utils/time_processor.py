def ms_to_string(milliseconds):
    """
    Formats milliseconds to hh:mm:ss:ms
    where:
    - hh is hours with leading zero if less than 10
    - mm is minutes with leading zero if less than 10
    - ss is seconds with leading zero if less than 10
    - ms is the two highest digits of the milliseconds
    """
    seconds = milliseconds // 1000
    milliseconds = int(milliseconds % 1000)
    minutes = int(seconds // 60)
    hours = int(minutes // 60)
    minutes = int(minutes % 60)
    seconds = int(seconds % 60)
    time_str = f"{hours:02}h:{minutes:02}m:{seconds:02}s:{milliseconds:03}ms"
    return time_str


def string_to_ms(time_str):
    """
    Parses a time string formatted as "hh:mm:ss:ms" back into total milliseconds.
    
    - hh is hours
    - mm is minutes
    - ss is seconds
    - ms is the two highest digits of the milliseconds
    
    Parameters:
        time_str (str): The time string to parse.
    
    Returns:
        int: The total number of milliseconds represented by the time string.
    """
    parts = time_str.split(':')
    hours = int(parts[0].strip('h'))
    minutes = int(parts[1].strip('m'))
    seconds = int(parts[2].strip('s'))
    milliseconds = int(parts[3].strip('ms'))

    total_milliseconds = (hours * 3600000) 
    total_milliseconds += (minutes * 60000)
    total_milliseconds += (seconds * 1000) 
    total_milliseconds += (milliseconds)

    return total_milliseconds
