import os
import platform

os_ = platform.system()

if os_ == 'Linux':
    pass
elif os_ == 'Windows':
    os.system("conda env create -f environment/environment_win64.yaml")
elif os_ == 'Darwin':
    pass