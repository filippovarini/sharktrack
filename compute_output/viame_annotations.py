import pandas as pd
from datetime import datetime

VIAME_COLUMNS = ['# 1: Detection or Track-id',	'2: Video or Image Identifier', '3: Unique Frame Identifier',	'4-7: Img-bbox(TL_x',	'TL_y',	'BR_x',	'BR_y)',	'8: Detection or Length Confidence',	'9: Target Length (0 or -1 if invalid)',	'10-11+: Repeated Species', 'Confidence Pairs or Attributes']

def max_conf2viame(max_conf_detections, fps, track_count):
  """
  Creates a csv file in VIAME format to upload the detection to the VIAME annotation
  software and clean the annotations.
"""
  data = []

  for index, row in max_conf_detections.iterrows():

    # Add row
    new_row = {
      '# 1: Detection or Track-id': row['track_id'], 
      '2: Video or Image Identifier': f"{row['track_id']}.jpg", 
      # instead of frame_id because VIAME maps annotation to image with the index in the imnage sequence 
      '3: Unique Frame Identifier': index + track_count,
      '4-7: Img-bbox(TL_x': row['xmin'], 
      'TL_y': row['ymin'], 
      'BR_x': row['xmax'], 
      'BR_y)': row['ymax'], 
      '8: Detection or Length Confidence': row['confidence'],
      '9: Target Length (0 or -1 if invalid)': -1,
      '10-11+: Repeated Species': row['class'],
      'Confidence Pairs or Attributes': row['confidence']
      }

    data.append(new_row)
    
  df = pd.DataFrame(data, columns=VIAME_COLUMNS)

  return df

def add_metadata_row(df, fps):
  # Add a line below the header with the values '# metadata'	'fps: 1'	'exported_by: "dive:python"', 'exported_time: "Mon Jan 15 15:56:15 2024"'	'Unnamed: 4'	'Unnamed: 5'	'Unnamed: 6'	'Unnamed: 7'	'Unnamed: 8'	'Unnamed: 9'	'Unnamed: 10'
  current_date = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
  metadata_row = {'# 1: Detection or Track-id': '# metadata', '2: Video or Image Identifier': f'fps: {fps}', '3: Unique Frame Identifier': 'exported_by: "Sharktrack User"', '4-7: Img-bbox(TL_x': f'exported_time: "{current_date}"', 'TL_y': 'Unnamed: 4', 'BR_x': 'Unnamed: 5', 'BR_y)': 'Unnamed: 6', '8: Detection or Length Confidence': 'Unnamed: 7', '9: Target Length (0 or -1 if invalid)': 'Unnamed: 8', '10-11+: Repeated Species': 'Unnamed: 9', 'Confidence Pairs or Attributes': 'Unnamed: 10'}
  df = pd.concat([df, pd.DataFrame([metadata_row])], ignore_index=True)
  # make sure metadata row is at the beginning of the df (index 0)
  df = df.reindex([len(df)-1] + list(range(len(df)-1)))
  # remove indices
  df = df.reset_index(drop=True)
  return df
   