import numpy as np
import os

files = os.walk('./')
for i in files:
    print(i, '\r\f')