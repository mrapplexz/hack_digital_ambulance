import re
import json

import numpy as np
import pandas as pd


def load_substations(substations_path):
    with open(substations_path, 'r') as f:
        subs = json.load(f)

    subs = pd.DataFrame(
        [{'name': name, 'lat': re.split(',? ', val)[0], 'lon': re.split(',? ', val)[1]} for name, val in subs.items() if
         name != 'NaN']).set_index('name')

    subs['lat'] = subs['lat'].astype(np.float64)
    subs['lon'] = subs['lon'].astype(np.float64)
    return subs