execfile('./initialization/initializer.py')

def all_data():
    print(dataset)
def data_summary():
    print(dataset.describe())

def plot_box_and_whisker():
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()

def plot_histogram():
    dataset.hist()
    plt.show()


def plot_multivariate_scatter():
    scatter_matrix(dataset)
    plt.show()
