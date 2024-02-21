# coding:utf-8
##******************************************************************************//
## * Project Director: Prof. Lining Zhang                                        //
## * Authors: Ying Ma, Yu Li.                                                    //
## * Notes: This version is utilized for run iMoB.                               //
## ******************************************************************************//


import subprocess

# Run imob open source.py
file_to_run = 'imob_Open_source.py'
subprocess.run(['python', file_to_run], check=True)