from numpy import array, diag, concatenate, savetxt, ones
from numpy.random import seed
from numpy.random import multivariate_normal

# just so that we get same values everytime
SEED_VALUE = 1024
seed(SEED_VALUE)

output_file = 'data.csv'

def generate_data():

    ## let us make nice clusters of points, well separated for a good
    ## demo

    ## We are creating 4 clusters of known mean and variance values
    ## each cluster will have 3000 data points

    means = [
        array([2.5, 3.5]),
        array([7.2, 8.3]),
        array([-2., -3]),
        array([-6.3, -9.2])
    ]

    covariances = [
        diag([0.5, 0.3]),
        diag([1.1, 1.2]),
        diag([1.5, 0.2]),
        diag([0.75, 1.2])
    ]

    data = [
        multivariate_normal(
            mean, scale, size=3000
        ) for mean, scale in zip(means, covariances)
    ]

    data = concatenate(data, axis=0)
    cluster = ones((3000*4, 1))
    cluster[:3000, 0] = 0
    cluster[3000:6000, 0] = 1
    cluster[6000:9000, 0] = 2
    cluster[9000:, 0] = 3
    data = concatenate((data, cluster), axis=1)
    savetxt(
        output_file,
        data,
        fmt="%f",
        delimiter=",",
        header="X,Y,Cluster"
    )

if __name__ == "__main__":
    generate_data()