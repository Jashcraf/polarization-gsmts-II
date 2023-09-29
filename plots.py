import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

def plot_stokes(stokes,normalized=False,cmap='inferno'):
    """plots stokes vector

    Parameters
    ----------
    stokes : list or ndarray with first dimension of length 4
        Stokes vector array
    """

    fig,ax = plt.subplots(ncols=4,figsize=[10,3])
    for i,data in enumerate(stokes):
        if normalized:
            data /= stokes[0]
        im = ax[i].imshow(data,cmap='inferno')
        div = make_axes_locatable(ax[i])
        cax = div.append_axes("right", size="7%", pad="2%")
        cb = fig.colorbar(im,cax=cax)
    plt.show()

