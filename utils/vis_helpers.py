import matplotlib.pylab as plt


def plot_grey_im(x, ax=None, title=''):
    if len(x.shape) > 3:
        x = x[0]
    if ax is not None:
        ax.imshow(x.transpose(1, 0).transpose(2, 1).detach(), cmap='gray');
        ax.tick_params(axis='both',          # changes apply to the x-axis
                       which='both',      # both major and minor ticks are affected
                       bottom=False,      # ticks along the bottom edge are off
                       top=False,         # ticks along the top edge are off
                       labelbottom=False, # labels along the bottom edge are off
                       right=False, left=False, labelleft=False)
        ax.set_title(title)
    else:
        plt.imshow(x.transpose(1, 0).transpose(2, 1).detach(), cmap='gray');
        plt.tick_params(axis='both',  # changes apply to the x-axis
                       which='both',  # both major and minor ticks are affected
                       bottom=False,  # ticks along the bottom edge are off
                       top=False,  # ticks along the top edge are off
                       labelbottom=False,  # labels along the bottom edge are off
                       right=False, left=False, labelleft=False)
        plt.title(title)