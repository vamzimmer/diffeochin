import sys
import tkinter
import matplotlib
# matplotlib.use('TkAgg')
i = 0
while i < 10:
    i += 1
    try:
        matplotlib.use('TkAgg')
        break
    except:
        # print(i)
        sys.stdout.write("\rTry using 'TKAgg' in matplotlib (%i)" % i)
        sys.stdout.flush()


from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
# from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy.spatial import ConvexHull
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import pyvista as pv
# pv.set_plot_theme("document")

from matplotlib.animation import FuncAnimation, PillowWriter
import logging
logging.getLogger('matplotlib.font_manager').disabled = True

camera_position = (42.46, 95.47, -14.41) # (34.16, 98.64, -25.20)
camera_focalpoint = (46.84, 56.82, 7.50) # (47.25, 56.75, 7.40,) #
camera_viewup = (-0.99, -0.03, 0.13) # (-0.97, -0.19, 0.14)
CAMERA = [camera_position, camera_focalpoint, camera_viewup]

# Nonlinear colormap
# https://stackoverflow.com/questions/22521382/nonlinear-colormap-matplotlib
class nlcmap(object):
    def __init__(self, cmap, levels):
        self.cmap = cmap
        self.N = cmap.N
        self.monochrome = self.cmap.monochrome
        self.levels = np.asarray(levels, dtype='float64')
        self._x = self.levels
        self.levmax = self.levels.max()
        self.transformed_levels = np.linspace(0.0, self.levmax,
             len(self.levels))

    def __call__(self, xi, alpha=1.0, **kw):
        yi = np.interp(xi, self._x, self.transformed_levels)
        return self.cmap(yi / self.levmax, alpha)


def get_convergence_values(log_file, conv_file):
    with open(log_file) as read_file:
        content = read_file.readlines()

    loglike = [s for s in content if "Log-like" in s]
    values = [x.split()[3] + ' ' + x.split()[7] + ' ' + x.split()[11][:-1] for x in loglike]

    with open(conv_file, "w") as text_file:
        for v in values:
            text_file.write("{}\n".format(v))


def plot_convergence(conv_file, out_dir, conv_img=None, ax=None, show=True):
    df = pd.DataFrame(data=np.loadtxt(conv_file, delimiter=' '), columns=['Log-Likelihood', 'Attachment', 'Regularity'])

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(8,6))
    ax.plot(df['Log-Likelihood'], 'r-', label='Log-Likelihood', linewidth=4.0)
    ax.plot(df['Attachment'], 'g--', label='Attachment', linewidth=4.0)
    ax.plot(df['Regularity'], 'b--', label='Regularity', linewidth=4.0)
    ax.set_xlabel('Iteration')
    ax.legend(bbox_to_anchor=(.6, .5), loc=2, borderaxespad=0., fancybox=True, framealpha=0.5)
    ax.grid(which='major')
    ax.set_title('Optimization convergence (log-likelihood)')
    if conv_img is None:
        conv_img = out_dir + "/convergence.png"
    if show:
        plt.show()
    plt.savefig(conv_img)
    plt.close()


def plot_shapes(shapes, titles=None, subplots=(1,1), pfile=None, show=True, colors=('red', 'green'), cmap=None, camera=None, line_width=5, window_size=None, is2d=False):

    """
    Parameters
    ----------
    shapes      (list of) shape(s); each list is plotted in the same subplot.
                shapes = shape : one subplot, one shape
                shapes = [shape_1, shape_2, ...] : one subplot, multiple shapes
                shapes = [[shape_1], [shape_2], ...] : multiple subplots, one shape
                shapes = [[shape_11, shape_12, ...], [shape_21, shape_22, ...], ...] : multiple subplots, multiples shapes
    titles      titles for each subplot
                optional, default None
    subplots    shape of subplots, e.g. (1,1), (2,3)
                optional, default (1,1)
    pfile       filename of png output file
                optional, default None
    show        if to show plot or not
                optional, default True
    colors      colors of shapes of each subplot
                optional, default ('red', 'green')

    Returns
    -------

    """

    if not isinstance(shapes, list):
        shapes = [shapes]
    if titles is not None and not isinstance(titles, list):
        titles = [titles]
    # if subplots is None:
    #     subplots = (1, 1)
    Nplots = subplots[0] * subplots[1]
    
    if window_size is not None:
        plotter = pv.Plotter(shape=subplots, window_size=window_size, off_screen=False if show else True)#, window_size=[4000, 4000])
    else:
        plotter = pv.Plotter(shape=subplots, off_screen=False if show else True)

    if titles is not None:
        assert len(shapes) <= len(titles), "Please provide at least as many title lists as shape lists."
    assert len(shapes) <= Nplots, 'Please provide at least as many subplots lists as shapes'

    for n in range(Nplots):
        sub_ind = np.unravel_index(n, (subplots[0], subplots[1]))
        plotter.subplot(sub_ind[0], sub_ind[1])

        if camera is not None:
            # tuple: camera location, focus point, viewup vector
            # plotter.camera_position = [(20.18, 124.50, -25.51), (47.25, 56.75, 7.40,), (-0.93, -0.27, 0.20)]
            plotter.camera_position = camera
        # else:
            # plotter.camera_position = CAMERA

        if titles is not None:
            plotter = subplot_shapes(plotter, shapes[n], title=titles[n], colors=colors, cmap=cmap, line_width=line_width, is2d=is2d)
        else:
            plotter = subplot_shapes(plotter, shapes[n], colors=colors, cmap=cmap, line_width=line_width, is2d=is2d)


    # if show:
    plotter.show(screenshot=pfile)
    # plotter.store_image = True
    # plotter.show()
    

    # if pfile:
    #     # save as png
    #     plt.imshow(plotter.image)
    #     plt.savefig(pfile, dpi = 300)
    #     plt.close()


def subplot_shapes(ax, shapes, title=None, colors=('red', 'green'), cmap=None, line_width=5, is2d=False):
    if not isinstance(shapes, list):
        shapes = [shapes]

    for n, mesh in enumerate(shapes):
        if title is not None:
            ax.add_text(title, font_size=15)
        if isinstance(mesh, str):
            mesh = pv.read(mesh)
        if cmap is None:
            ax.add_mesh(mesh, color=colors[n], line_width=line_width)
        else:
            ax.add_mesh(mesh, cmap=cmap, line_width=line_width)
        if is2d:
            ax.view_xy()  # if mesh_2D is on the xy plane.

    return ax




def create_gif(data_dir, out_gif, camera_position=None, line_width=7, is2d=False, clim=None):

    if isinstance(data_dir, list):
        files = data_dir
    else:
        files = sorted(glob.glob('{}/*'.format(data_dir), recursive=True))

    plotter = pv.Plotter(notebook=False, off_screen=True)
    # pv.set_plot_theme("document")
    # plotter.add_mesh(mean, smooth_shading=False)

    # Open a gif
    plotter.open_gif(out_gif)
    if camera_position is not None:
        # tuple: camera location, focus point, viewup vector
        # plotter.camera_position = [(20.18, 124.50, -25.51), (47.25, 56.75, 7.40,), (-0.93, -0.27, 0.20)]
        plotter.camera_position = camera_position

    for mesh in files:
        # actor = plotter.add_mesh(pv.read(mesh), smooth_shading=True, cmap='coolwarm', line_width=line_width)
        if is2d:
            # rotate midsagittal profile, so that is has the same orientation as the symphyseal surface
            rotz = pv.read(mesh).rotate_z(270, inplace=False)
            rot = rotz.rotate_y(180, inplace=False)
            actor = plotter.add_mesh(rot, smooth_shading=True, cmap='coolwarm', line_width=line_width, clim=clim)
            plotter.view_xy()
        else:
            actor = plotter.add_mesh(pv.read(mesh), smooth_shading=True, cmap='coolwarm', line_width=line_width, clim=clim)
        # add anterior and posterior information
        plotter.add_text('p', position=(60,350), color='gray')
        plotter.add_text('a', position=(920,350), color='gray')
        plotter.add_text('a: anterior; p: posterior', position='upper_left', color='gray')

        plotter.mesh.compute_normals(cell_normals=False, inplace=True)
        plotter.render()
        plotter.write_frame()

        plotter.remove_actor(actor)
    # Closes and finalizes movie
    plotter.close()


def scatter_classes(X, X_classes, X_fun, classes, title, ids=None, img=None, colors=None, markers=None, colors2=None, limits=None, ax=None, fig=None, showit=True, axis_label=None, cmap=None):
    no_dims = X.shape[1]
    N = X.shape[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    h = []
    patches = []
    if colors is None:
        colors = ['b', 'r', 'g', 'm', 'y', 'b', 'k']
    if colors2 is None:
        colors2 = np.array([.5, 6, 3, .8, 4, 2, 1])
        # colors2 = np.array([6, .5, 3, .8, 4, 2, 1])
    if markers is None:
        markers = ['o', '^', 's', 'd', 'P', 'X', 'P']
    ms = 50
    if ids is None:
        ms = 100
    elif None in ids:
        ms = 100
    if axis_label is None:
        axis_label = ['PC1', 'PC2']

    if X_fun.any():
        _min, _max = np.amin([x for x in X_fun if str(x) != 'nan']), np.amax([x for x in X_fun if str(x) != 'nan'])
        # print(_min)
        # print(_max)

    # if cmap is None:
        # cmap = matplotlib.cm.jet

    for j in range(0, len(classes)):

        ind = (np.where(X_classes == j))[0]
        Y = X[ind, :2]
        # if ids is not None:
        #     idsy = [ids[j] for j in ind]
        #     # idsy = ids[ind]

        if X_fun.any():
            Y_fun = X_fun[ind]

        if not colors[j]=='k':
            if len(ind) > 2:

                try:
                    hull = ConvexHull(Y)
                    vertices = np.zeros((len(hull.vertices), 2))
                    vertices[:, 0] = Y[hull.vertices, 0]
                    vertices[:, 1] = Y[hull.vertices, 1]
                    polygon = Polygon(vertices, True)
                    patches.append(polygon)
                except scipy.spatial.qhull.QhullError:
                    print("QhullError: Data might lie on a line...")
                    ax.plot(Y[:, 0], Y[:, 1], color=colors[j], alpha=0.2)

            else:
                ax.plot(Y[:, 0], Y[:, 1], color=colors[j], alpha=0.2)

        if not X_fun.any():
            scat = ax.scatter(Y[:, 0], Y[:, 1], color=colors[j], label=classes[j], marker=markers[j], s=ms)
            h.append(scat)
        else:
            ind1 = (np.where(np.isnan(Y_fun)))[0]
            scat = ax.scatter(Y[ind1, 0], Y[ind1, 1], color="k", marker=markers[j], label=classes[j],
                               facecolors='none', s=ms)
            h.append(scat)

            ind2 = (np.where(~np.isnan(Y_fun)))[0]
            if cmap is None:
                scat2 = ax.scatter(Y[ind2, 0], Y[ind2, 1], c=Y_fun[ind2], cmap=matplotlib.cm.jet, vmin=_min, vmax=_max,
                                    label=classes[j],
                                    marker=markers[j], s=ms)
                h.append(scat2)
                if j == 0:
                    plt.colorbar(scat2, ax=ax, cmap=matplotlib.cm.jet)#, vmin=np.min(X_fun), vmax=np.max(X_fun))
            else:
                # Nonlinear colorbar
                scat2 = ax.scatter(Y[ind2, 0], Y[ind2, 1], edgecolors=cmap(Y_fun[ind2]), c=cmap(Y_fun[ind2]), s=ms,
                                    vmin=_min, vmax=_max,
                                    label=classes[j],
                                    marker=markers[j])
                h.append(scat2)
                if j == 0:
                    fig.subplots_adjust(left=0.21)
                    cbar_ax = fig.add_axes([0.1, 0.11, 0.015, 0.75])

                    #for the colorbar we map the original colormap, not the nonlinear one:
                    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, 
                                    norm=plt.Normalize(vmin=0, vmax=np.nanmax(X_fun)))
                    sm._A = []

                    cbar = fig.colorbar(sm, cax=cbar_ax)
                    #here we are relabel the linear colorbar ticks to match the nonlinear ticks
                    cbar.set_ticks(cmap.transformed_levels)
                    cbar.set_ticklabels(["%.0f" % lev for lev in cmap.levels])

        if ids is not None:
            for i in range(len(ind)):
                # print(i,ind[i], ids[ind[i]], idsy[i])
                if ids[ind[i]] is not None:
                    ax.text(Y[i, 0], Y[i, 1], '%s' % ids[ind[i]], size=8, zorder=1, color='k')

    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.2)
    p.set_clim([0, 5])
    p.set_array(np.array(colors2))
    ax.set_title(title)
    ax.add_collection(p)

    # if use_legend:
    # ax.legend(handles=h, loc='best', fontsize=14)#, bbox_to_anchor=(1, 0.5))
    ax.legend(handles=h, loc='best', fontsize=5)#, bbox_to_anchor=(1, 0.5))
    
    labelsx = np.round(np.linspace(limits[0], limits[1], 5), decimals=1)
    labelsy = np.round(np.linspace(limits[2], limits[3], 5), decimals=1)
    ax.set_xticks(labelsx)
    ax.set_yticks(labelsy)
    ax.set_xticklabels(labelsx, fontsize=14)
    ax.set_yticklabels(labelsy, fontsize=14)
    ax.set_xlabel(axis_label[0], fontsize=14)
    ax.set_ylabel(axis_label[1], fontsize=14, labelpad=-3)
    # 
    if limits is not None:
        # set the limits
        ax.set_xlim(limits[:2])
        ax.set_ylim(limits[2:])
    else:
        ax.axis('equal')

    if img is not None:
        plt.savefig(img, dpi = 300)

    if showit:
        plt.show()


def scatter_classes3(X, X_classes, classes, fig=None, ax=None, ids=None, title=None, colors=None, markers=None, pfile=None, showit=True):
    no_dims = X.shape[1]
    N = X.shape[0]

    h = []
    patches = []
    if colors is None:
        colors = ['b', 'r', 'g', 'm', 'y', 'b', 'k']
    colors2 = np.array([.5, 6, 3, .8, 5, 2, 1])
    if markers is None:
        markers = ['o', '^', 's', 'd', 'P', 'X', 'P']

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(1, ax, 1, projection='3d')
    

    # fig, ax = plt.subplots(figsize=(10, 8))

    for j in range(0, len(classes)):

        ind = (np.where(X_classes == j))[0]
        Y = X[ind, :3]

        if not colors[j]=='k':
            if len(ind) > 3:
                hull = ConvexHull(Y)

                for i in range(0, hull.simplices.shape[0]):
                    simplex = hull.simplices[i, :]
                    ax.plot_trisurf(Y[simplex, 0], Y[simplex, 1], Y[simplex, 2], color=colors[j], alpha=0.2)
            elif len(ind) == 3:
                ax.plot_trisurf(Y[:, 0], Y[:, 1], Y[:, 2], color=colors[j], alpha=0.2)
            elif len(ind) == 2:
                ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], color=colors[j], alpha=0.2)

        scat = ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color=colors[j], label=classes[j], marker=markers[j])
        h.append(scat)

        if ids is not None:
            for i in range(len(ind)):
                ax.text(Y[i, 0], Y[i, 1], Y[i, 2], '%s' % (ids[ind[i]]), size=7, zorder=1, color='k')

    ax.legend(handles=h, loc='best', fontsize=8) # loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    if pfile is not None:
        plt.savefig(pfile, dpi = 300)
    
    if showit:
        plt.show()


def plot_vector(X, vector, group1, group2, group_ids, group_names, pfile=None, markers1=None, markers2=None, colors1=None, colors2=None, limits=None):
    if markers1 is None:
        markers1 = ['o', '^']
    if markers2 is None:
        markers2 = ['s', 'd']
    if colors1 is None:
        colors1 = ['red', 'salmon']
    if colors2 is None:
        colors2 = ['blue', 'cornflowerblue']

    fig, ax = plt.subplots(figsize=(10, 8))

    ids1 = group_ids[group1]
    ids1 = list(set(ids1))
    for j in range(len(ids1)):
        ind = (np.where(group_ids == ids1[j]))[0]
        plt.scatter(X[ind, 0], X[ind, 1], c=colors1[j], s=150, marker=markers1[j],
                    label=group_names[0][j])  # edgecolor='k')
    ids2 = group_ids[group2]
    ids2 = list(set(ids2))
    for j in range(len(ids2)):
        ind = (np.where(group_ids == ids2[j]))[0]
        plt.scatter(X[ind, 0], X[ind, 1], c=colors2[j], s=150, marker=markers2[j],
                    label=group_names[1][j])  # edgecolor='k')

    plt.quiver([vector[0] / 2, vector[0] / 2],  # [0, 0],
               [vector[1] / 2, vector[1] / 2],  # [0, 0],
               [.5 * vector[0], -.5 * vector[0]],
               [.5 * vector[1], -.5 * vector[1]],
               color=cm.jet([60, 220]),
               scale=1.5, edgecolor='k', linewidth=2, width=0.012)

    # ax.legend(handles=h, loc='best', fontsize=5)#, bbox_to_anchor=(1, 0.5))
    
    labelsx = np.round(np.linspace(limits[0], limits[1], 5), decimals=1)
    labelsy = np.round(np.linspace(limits[2], limits[3], 5), decimals=1)
    plt.xticks(labelsx, fontsize=14)
    plt.yticks(labelsy, fontsize=14)
    # plt.xticklabels(labelsx, fontsize=14)
    # plt.yticklabels(labelsy, fontsize=14)
    plt.xlabel('PC 1', fontsize=11)
    plt.ylabel('PC 2', fontsize=11)
    # 
    if limits is not None:
        # set the limits
        plt.xlim(limits[:2])
        plt.ylim(limits[2:])
    else:
        plt.axis('equal')

    plt.legend(loc='best', fontsize=5)
    plt.colorbar()

    #     plt.grid(True)
    if pfile is not None:
        plt.savefig(pfile, dpi = 300)

class ScatterClasses3Animation(object):
    """
    Creates a 3d plot.

    Usage:

    animator = pltutils.ScatterClasses3Animation(X=X_kpca, X_classes=tax_ids, X_fun=np.asarray([]), classes=groups,
                                             ids=spec, title='Taxonomy')

    # just static 3d
    animator.draw()

    # automatic rotation of 3d plot
    animator.init_animation()
    animator.rotate()

    # creation of animation of rotating 3d plot
    animator.create_animation(kpca_dir+'/kpca.gif', fps=10)

    """
    def __init__(self, X, X_classes, X_fun, classes, title, ids=None, axis_label=None, colors=None, markers=None):
        self.X = X
        self.X_classes = X_classes
        self.X_fun = X_fun
        self.classes = classes
        self.ids = ids
        self.title = title
        self.axis_label = axis_label

        self.colors = colors
        if colors is None:
            self.colors = ['b', 'r', 'g', 'm', 'y', 'b', 'k']
        self.colors2 = np.array([.5, 6, 3, .8, 5, 2, 1])
        self.markers = markers
        if markers is None:
            self.markers = ['o', '^', 's', 'd', '^', 'X', 'P']

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def draw(self, show=True):
        no_dims = self.X.shape[1]
        N = self.X.shape[0]

        h = []
        patches = []
        

        for j in range(0, len(self.classes)):

            ind = (np.where(self.X_classes == j))[0]
            Y = self.X[ind, :3]

            if not self.colors[j]=='k':
                if len(ind) > 3:
                    hull = ConvexHull(Y)

                    for i in range(0, hull.simplices.shape[0]):
                        simplex = hull.simplices[i, :]
                        self.ax.plot_trisurf(Y[simplex, 0], Y[simplex, 1], Y[simplex, 2], color=self.colors[j], alpha=0.2)
                elif len(ind) == 3:
                    self.ax.plot_trisurf(Y[:, 0], Y[:, 1], Y[:, 2], color=self.colors[j], alpha=0.2)
                elif len(ind) == 2:
                    self.ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], color=self.colors[j], alpha=0.2)

            scat = self.ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color=self.colors[j], label=self.classes[j], marker=self.markers[j])
            h.append(scat)

            if self.ids is not None:
                for i in range(len(ind)):
                    self.ax.text(Y[i, 0], Y[i, 1], Y[i, 2], '%s' % (self.ids[ind[i]]), size=7, zorder=1, color='k')

        plt.legend(handles=h, loc='best')# loc='center left', bbox_to_anchor=(1, 0.5))
        self.ax.set_xlabel(self.axis_label[0])
        self.ax.set_ylabel(self.axis_label[1])
        self.ax.set_zlabel(self.axis_label[2])
        #self.ax.axis('equal')

        if show:
            plt.show()

    def init_animation(self):
        self.draw(False)

    def rotate(self):

        for angle in range(0, 360, 3):
            self.ax.view_init(30, angle)
            plt.draw()
            plt.pause(.01)

        for angle in range(0, 360, 3):
            self.ax.view_init(angle, 30)
            plt.draw()
            plt.pause(.01)

    def update_animation(self, angle):
        if angle <= 360:
            self.ax.view_init(30, angle)
            plt.draw()
            # plt.pause(.0001)
        else:
            self.ax.view_init(angle-360, 30)
            plt.draw()
            # plt.pause(.0001)

    def create_animation(self, filename, fps=15, view=np.linspace(0, 720, 250)):
        ani = FuncAnimation(self.fig, self.update_animation, view, init_func=self.init_animation)

        writer = PillowWriter(fps=fps)
        ani.save(filename, writer=writer)

        # plt.show()



