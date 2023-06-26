# -*- coding: utf-8 -*-
"""
Spyder Editor
Author: Marc Botifoll Moral
All rights reserved
"""

import sys
import subprocess





def install_pip():
    
    list_of_packages=['numpy', 'matplotlib', 'scikit-learn', 'scikit-image', 'scipy', 'opencv-python', 'hyperspy', 'Pillow','tk','stemtool','dask[distributed]']
     
    for package in list_of_packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    
    #check if installation done   
    reqs=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])  
    list_all_versions=[r.decode().split() for r in reqs.split()]
    print(list_all_versions)
    
    
    
def install_conda():
    
    list_of_packages=['numpy', 'matplotlib', 'scikit-learn', 'scikit-image', 'scipy', 'opencv-python', 'hyperspy', 'Pillow','tk','stemtool','dask[distributed]']

    for package in list_of_packages:
        subprocess.check_call([sys.executable, '-m', 'conda', 'install', package])
        
    #check if installation done
    reqs=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])    
    list_all_versions=[r.decode().split() for r in reqs.split()]
    print(list_all_versions)


install_pip()
