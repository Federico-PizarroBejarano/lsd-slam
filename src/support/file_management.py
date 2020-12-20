import datetime

def get_filename(files, frame):
    """
    Get the filename of an image with the corresponding frame number

    Parameters
    ----------
    files (list): a list of all filenames in folder
    frame (int): the desired frame number
    
    Returns
    -------
    filename (str): the filename (no path)
    """

    frame_str = str(frame).zfill(6)
    corresponding_files = [image for image in files if f'frame{frame_str}' in image]

    return corresponding_files[0]


def get_unix_timestamp(filename):
    """
    Get a unix timestamp in seconds from an image filename
    
    Parameters
    ----------
    filename (str): an image filename with date and time (Toronto time zone) in the
        'YYYY_MM_DD_hh_mm_ss_microsec' format
    
    Returns
    -------
    float: equivalent unix timestamp, in seconds
    """

    date_and_time = filename.split('frame')[1][7:-4]
    
    # Extract microseconds part and convert it to seconds
    microseconds_str = date_and_time.split('_')[-1]
    seconds_remainder = float('0.' + microseconds_str.zfill(6))
    
    # Extract date without microseconds and convert to unix timestamp
    # The added 'GMT-0400' indicates that the provided date is in the 
    # Toronto (eastern) timezone during daylight saving
    date_to_sec_str = date_and_time.replace('_' + microseconds_str, '')
    epoch = \
        datetime.datetime.strptime(date_to_sec_str + \
        ' GMT-0400', "%Y_%m_%d_%H_%M_%S GMT%z").timestamp()
    
    # Add the microseconds remainder
    epoch = epoch + seconds_remainder

    return epoch