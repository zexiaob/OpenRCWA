# Author: Jordan Edmunds, Ph.D. Student, UC Berkeley
# Contact: jordan.e@berkeley.edu
# Creation Date: 11/01/2019
#
from rcwa import Material, Layer, LayerStack, Source, Solver
import numpy as np
import warnings


def solve_system():
        startWavelength = 0.25
        stopWavelength = 0.85
        stepWavelength = 0.001

        source = Source(wavelength=startWavelength)
        si = Material(name='Si')

        reflectionLayer = Layer(n=1) # Free space
        thin_film = Layer(thickness=0.1, material=si)
        transmissionLayer = Layer(n=4)
        stack = LayerStack(thin_film, incident_layer=reflectionLayer, transmission_layer=transmissionLayer)

        print("Solving system...")
        TMMSolver = Solver(stack, source, (1, 1))
        wavelengths = np.arange(startWavelength, stopWavelength + stepWavelength,
                stepWavelength)

        results = TMMSolver.solve(wavelength=wavelengths)
        warnings.warn("example warning")
        return results


if __name__ == '__main__':
        results = solve_system()
        import matplotlib.pyplot as plt
        plt.show()
