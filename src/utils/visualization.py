import matplotlib.pyplot as plt

def plot_predictions(test_labels, test_predictions):
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True values (expenses)')
    plt.ylabel('Predictions (expenses)')
    lims = [0, 50000]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims,lims)