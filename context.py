"""
Adds the source files to the path for files in any subdirectory
TODO: check that we have not alredy added to our path.
"""
import os
import sys

fileLocation = os.path.dirname(os.path.abspath(__file__))
sourceLocation = os.path.abspath(os.path.join(fileLocation, 'RCWA/source/'))
nkLocation = os.path.abspath(os.path.join(fileLocation, 'RCWA/nkData/'))
netlistLocation = os.path.abspath(os.path.join(fileLocation, 'RCWA/netlist/'))
testLocation = os.path.abspath(os.path.join(fileLocation, 'RCWA/test/'))

sys.path.insert(0, sourceLocation)
sys.path.insert(0, nkLocation)
sys.path.insert(0, netlistLocation)
sys.path.insert(0, testLocation)
