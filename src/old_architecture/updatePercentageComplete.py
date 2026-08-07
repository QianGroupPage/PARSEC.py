import matplotlib.pyplot as plt


def updatePercentageComplete(percentageComplete, barHandle, handles):
    """
    Update the progress bar in the graphical user interface.

    Args:
        percentageComplete (float): Current percentage of completion (0 to 1).
        barHandle: Handle to the progress bar (matplotlib object).
        handles: Dictionary of handles to GUI elements.
    """
    barHandle.set_height(percentageComplete)
    plt.draw()
    plt.pause(0.001)
