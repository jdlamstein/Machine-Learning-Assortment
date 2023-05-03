import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kstest


def read_stack(file, debug=False):
    stack = imageio.v3.imread(file)
    if debug:
        plt.imshow(stack[0])
        plt.show()
    return stack


def data_stats(stack, savepath=None, plotbool=False):
    uni, cnts = np.unique(stack, return_counts=True)
    print(f'Max: {max(uni)}, Min: {min(uni)}')
    print(f'Total counts {sum(cnts)}')
    # get intensity distribution
    if plotbool:
        plt.bar(uni, cnts)
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Counts')
        plt.title('Pixel Intensity Histogram')
        plt.savefig(savepath)
        plt.show()
    return uni, cnts

def check_normality(arr):
    # Shapiro test has greater power than the Kolmogorov-Smirov Test, but KS test_2 can test_2 against multiple distributions.
    shapiro_result = shapiro(arr)
    ks_result = kstest(arr, 'norm')
    #Shapiro-Wilk test struggles if the distribution has many identical values, like our distribution.
    print(shapiro_result)
    print(ks_result)


if __name__ == '__main__':
    f = '/Users/gandalf/PycharmProjects/Segmentation/resources/take_home_movie_compresed.tiff'
    histogram_savepath = '/Users/gandalf/PycharmProjects/Segmentation/figures/histogram.png'
    stack = read_stack(f)
    uni, cnts = data_stats(stack, histogram_savepath, plotbool=False)
    check_normality(cnts)
