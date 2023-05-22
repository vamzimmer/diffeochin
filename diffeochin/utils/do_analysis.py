import os
from sre_parse import fix_flags
import numpy as np
import pandas as pd
from numpy import linalg as LA
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn import manifold
from sklearn import preprocessing as preprocessing
from IPython.display import Image

import diffeochin.utils.class_utils as cutils
import diffeochin.utils.eval_utils as eutils
import diffeochin.utils.stat_utils as sutils
import diffeochin.utils.plot_utils as pltutils

'''
    Do the statistical analysis
'''

def load_atlas_data(atlas_dir, df):
    '''
        Load momenta etc from deterministic atlas
    '''
    controlpoints = np.loadtxt(atlas_dir + '/DeterministicAtlas__EstimatedParameters__ControlPoints.txt')
    f = open(atlas_dir + '/DeterministicAtlas__EstimatedParameters__Momenta.txt')
    first_line = f.readline().split(' ')
    number_of_subjects = int(first_line[0])
    number_of_controlpoints = int(first_line[1])
    dimension = int(first_line[2])
    f.close()
    momenta = np.loadtxt(atlas_dir + '/DeterministicAtlas__EstimatedParameters__Momenta.txt', skiprows=2)
    momenta_linearised = momenta.reshape([number_of_subjects, dimension*number_of_controlpoints])

    momenta_linearised2 = momenta.reshape([number_of_subjects, number_of_controlpoints, dimension])
    momenta_norm = [np.mean(LA.norm(momenta_linearised2[j, :, :], axis=1)) for j in range(number_of_subjects)]
    df['momenta_norm'] = momenta_norm
    np.savetxt(atlas_dir+'/momentas.csv',momenta_linearised,fmt='%.5f', delimiter=',')

    print('Control Points: {}'.format(number_of_controlpoints))
    print('Subjects: {}'.format(number_of_subjects))
    print('Dimension: {}'.format(dimension))

    print(momenta_linearised.shape)

    return momenta_linearised


def load_distance_matrix(out_dir, reg_dir, df, kernel=None):
    if kernel is not None:
        distance_file = out_dir + '/D.xlsx'
    else:
        distance_file = out_dir + '/D_rbf.xlsx'

    D = eutils.read_data(distance_file, sheet='distance_matrix')
    D = D.drop(columns=['Unnamed: 0'])
    # compute norm of momentas
    df['momenta_norm'] = (D.sum().to_numpy()[1:]/(len(df)-1)).astype(np.float)

    return D

def perform_kpca(X, df, n_components, precomputed=False):
    '''
        Perform kernel PCA on
            {1} atlas momenta -> precomputed=False
            {2} distance matrix -> precomputed=True
    '''
    gam = 0.01 #0.01
    PCs = ['PC{}'.format(n+1) for n in range(n_components)]

    if not precomputed:
        kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, n_components=n_components, gamma=gam)
    else:
        kpca = KernelPCA(kernel="precomputed", n_components=n_components)
    X_kpca = kpca.fit_transform(X)
    for idx in range(X_kpca.shape[1]):
        df['PC{}'.format(idx+1)] = X_kpca[:, idx]
        # absolute coordinate value
        df['abs_PC{}'.format(idx + 1)] = abs(df['PC{}'.format(idx + 1)])

    eigenvalues = kpca.lambdas_
    eigenvectors = kpca.alphas_ / np.linalg.norm(kpca.alphas_, axis=0)

    eig = pd.DataFrame()
    eig['PCA dimension'] = ['PC{}'.format(idx+1) for idx in range(len(eigenvalues))]
    eig['cum. variability (in %)'] = 100 * np.cumsum(eigenvalues) / np.sum(eigenvalues)
    pd.set_option('display.precision', 2)
    eig['lambda'] = eigenvalues
    eig['alpha'] = list(np.transpose(eigenvectors))

    # z scores
    compute_zscores(df, n_components, abs=True)

    return eig


def perform_mds(D, df, n_components):
    mds = manifold.MDS(n_components=n_components, max_iter=3000, eps=1e-9, random_state=1, dissimilarity="precomputed", n_jobs=None)
    X_mds = mds.fit(D).embedding_
    for idx in range(X_mds.shape[1]):
        df['PC{}'.format(idx + 1)] = X_mds[:, idx]
        # absolute coordinate value
        df['abs_PC{}'.format(idx + 1)] = abs(df['PC{}'.format(idx + 1)])

    # z scores
    compute_zscores(df, n_components, abs=True)


def compute_zscores(df, n_components, abs=False):

    PCs = ['PC{}'.format(n+1) for n in range(n_components)]
    absPCs = ['abs_PC{}'.format(n+1) for n in range(n_components)]

    # compute z-scores
    scaler = preprocessing.StandardScaler()
    age = df['age_in_days'].tolist()
    agez = scaler.fit_transform(df['age_in_days'].to_numpy().reshape(-1, 1))
    df['age_zscore'] = agez
    momentaz = scaler.fit_transform(df['momenta_norm'].to_numpy().reshape(-1, 1))
    df['momenta_norm_zscore'] = momentaz

    PCz = scaler.fit_transform(df[PCs])
    for i in range(n_components):
        df['PC{}_zscore'.format(i+1)] = PCz[:, i]
    
    if abs:
        PCz = scaler.fit_transform(df[absPCs])    
        for i in range(n_components):
            df['abs_PC{}_zscore'.format(i+1)] = PCz[:, i]


def taxon_classification(df, subset, n_components, outfile, merge=False, showit=True):
    print('Taxon Classification')
    '''
    Logistic regression on Taxon
        1.1 Logistic regression with Cross-validation (LOO)
        1.2 ANOVA + Simple logistic regression with most predictive coordinate
        1.3 Magnitude of variation between groups (within and between group variation)
    '''

    fig, ax = plt.subplots(1, 1 if not merge else 2, figsize=(10, 4))
    PCs = ['PC{}_zscore'.format(n+1) for n in range(n_components)]
    # PCs = ['PC{}'.format(n+1) for n in range(n_components)]

    X = df[PCs]
    y = df['Group_ids']

    # 1.1 Logistic regression with Cross-validation (LOO)
    print('Logistic regression with Cross-validation')
    preds, ptest = sutils.my_logistic_regression(X, y)
    res = cutils.evaluation(y, preds, subset, outfile, 'LogReg_CV_ncomp{}'.format(n_components), df['specimen'], ax=None if not merge else ax[0], permutation_test=ptest, showit=showit)
    

    if merge:
        # merge modern humans and merge hominins
        yy_set = list(set(y))
        subs = ['modern_humans', 'fossils']
        yy = [0 if j<2 else 1 for j in y]

        df['yy'] = np.array(yy)
        preds, ptest = sutils.my_logistic_regression(X, yy)
        res = cutils.evaluation(yy, preds, subs, outfile, 'LogReg_CV', df['specimen'], ax=ax[1], permutation_test=ptest)
    print('Done')
    
    # 1.2 ANOVA + Simple logistic regression with most predictive coordinate
    y = 'Group_ids'
    X = ' + '.join(PCs)
    pfile = outfile.replace('.xlsx', '_SLR')

    print('ANOVA + Simple logistic regression with most predictive coordinate')
    anova_table, slr = sutils.my_anova_logistic(df, X, y, pfile, showit=showit)

    cutils.df_save_to_excel(outfile, anova_table, 'LogReg_ANOVA_ncomp{}'.format(n_components))
    cutils.df_save_to_excel(outfile, slr, 'LogReg_SLR_ncomp{}'.format(n_components))
    print('Done')

    # 1.3 Magnitude of variation between groups (within and between group variation)
    print('Magnitude of variation between groups')
    
    PCs = ['PC{}'.format(n+1) for n in range(n_components)]
    df2 = sutils.group_variations(df, PCs, 'Group_names')
    cutils.df_save_to_excel(outfile, df2, 'ANOVA_variation_ncomp{}'.format(n_components))
    print('Done')

    
def gender_classification(df, subset, subset_gender, n_components, outfile, showit=True):
    print('Gender Classification')
    '''
    Logistic regression on Taxon
        1.1 Logistic regression with Cross-validation (LOO)
        1.2 ANOVA + Simple logistic regression with most predictive coordinate
        1.3 Magnitude of variation between groups (within and between group variation)
    '''
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    PCs = ['PC{}_zscore'.format(n+1) for n in range(n_components)]
    # PCs = ['PC{}'.format(n+1) for n in range(n_components)]

    df_gen = df.loc[df['Group_names_2'].isin(subset_gender)]
    X = df_gen[PCs]
    y = df_gen['sexEnc']
    subs = ['Male', 'Female']

    # 1.1 Logistic regression with Cross-validation (LOO)
    print('Logistic regression with Cross-validation')
    preds, ptest = sutils.my_logistic_regression(X, y)
    res = cutils.evaluation(y, preds, subs, outfile, 'LogReg_CV_ncomp{}'.format(n_components), df_gen['specimen'], permutation_test=ptest, showit=showit, ax=ax[0])
    # European and African separated
    for si, sub in enumerate(subset[:2]):
        subs = subset_gender[si*2:si*2+2]
        df_gen = df.loc[df['Group_names_2'].isin(subs)]
        X = df_gen[PCs]
        y = df_gen['sexEnc']

        preds, ptest = sutils.my_logistic_regression(X, y)
        res = cutils.evaluation(y, preds, subs, outfile, 'LogReg_CV_ncomp{}'.format(n_components), df_gen['specimen'], permutation_test=ptest, ax=ax[si+1])
    print('Done')

    # 1.2 ANOVA + Simple logistic regression with most predictive coordinate
    y = 'sexEnc'
    X = ' + '.join(PCs)
    pfile = outfile.replace('.xlsx', '_SLR.png')

    fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4))
    colors = ['b', 'deepskyblue', 'r', 'orange']
    markers = ['o', 'o', '^', '^']

    df_gen = df.loc[df['Group_names_2'].isin(subset_gender)]
    print('ANOVA + Simple logistic regression with most predictive coordinate')
    anova_table, slr = sutils.my_anova_logistic(df_gen, X, y, pfile, ax=ax2[0], group_names='Group_names_2', colors=colors, markers=markers, showit=showit)
    cutils.df_save_to_excel(outfile, anova_table, 'LogReg_ANOVA_ncomp{}'.format(n_components))
    cutils.df_save_to_excel(outfile, slr, 'LogReg_SLR_ncomp{}'.format(n_components))

    for si, sub in enumerate(subset):
        subs = subset_gender[si*2:si*2+2]
        cols = colors[si*2:si*2+2]
        mks = markers[si*2:si*2+2]
        df_gen = df.loc[df['Group_names_2'].isin(subs)]

        anova_table, slr = sutils.my_anova_logistic(df_gen, X, y, pfile, ax=ax2[si+1], group_names='Group_names_2', colors=cols, markers=mks, showit=showit)
        cutils.df_save_to_excel(outfile, anova_table, '{}_LogReg_ANOVA_ncomp{}'.format(sub,n_components))
        cutils.df_save_to_excel(outfile, slr, '{}_LogReg_SLR_ncomp{}'.format(sub,n_components))
    print('Done')

    # 1.3 Magnitude of variation between groups (within and between group variation)
    print('Magnitude of variation between groups')
    
    PCs = ['PC{}'.format(n+1) for n in range(n_components)]
    df2 = sutils.group_variations(df, PCs, 'Group_names_2')
    cutils.df_save_to_excel(outfile, df2, 'ANOVA_variation_ncomp{}'.format(n_components))
    df2 = sutils.group_variations(df, PCs, 'sexEnc')
    cutils.df_save_to_excel(outfile, df2, 'ANOVA_variation_MF_ncomp{}'.format(n_components))
    print('Done')


def age_regression(df, n_components, outfile, showit=True, zscore=True):

    PCs = ['PC{}_zscore'.format(n+1) for n in range(n_components)]
    y = 'age_zscore'
    pfile = outfile.replace('.xlsx', '_SLR.png')
    if not zscore:
        PCs = ['PC{}'.format(n+1) for n in range(n_components)]
        # y = 'age_in_days'
        y = 'age_in_years'

    X = ' + '.join(PCs)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    colors = ['b', 'r', 'k', 'k', 'k']
    markers = ['o', '^', 'P', 'P', '+']

    # only European vs. African
    anova_table, slr, pc = sutils.my_anova_linear(df, X, y, pfile, ax=ax[0], group_names='Group_names', colors=colors, markers=markers, showit=showit)

    # European vs. African, Male vs. Female
    colors = ['b', 'deepskyblue', 'k', 'r', 'orange', 'k', 'k', 'k','k']
    markers = ['o', 'o', 'o', '^', '^', '^', 's', 'd', 'P']
    anova_table, slr, _ = sutils.my_anova_linear(df, X, y, pfile, ax=ax[1], group_names='Group_names_2', colors=colors, markers=markers, showit=showit)

    # non-linear regression
    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4))
    colors = ['b', 'r', 'k', 'k', 'k']
    markers = ['o', '^', 'P', 'P', '+']
    pfile2 = pfile.replace('.png', '_NL.png')
    r2, r2_all = sutils.my_nonlinear_regression(df, X, y, pc, pfile2, ax=ax2, group_names='Group_names', colors=colors, markers=markers, showit=showit)

    slr['R squared (rbf)'] = r2[0]
    slr['R squared (poly)'] = r2[1]
    slr['R squared (lin)'] = r2[2]
    slr['R squared (rbf, all PCs)'] = r2_all[0]
    slr['R squared (poly, all PCs)'] = r2_all[1]
    slr['R squared (lin, all PCs)'] = r2_all[2]

    cutils.df_save_to_excel(outfile, anova_table, 'LinReg_ANOVA_ncomp{}'.format(n_components))
    cutils.df_save_to_excel(outfile, slr, 'LinReg_SLR_ncomp{}'.format(n_components))



def morph_regression(df, n_components, outfile, abs=False, showit=True):

    if not abs:
        PCs = ['PC{}_zscore'.format(n+1) for n in range(n_components)]
    else:
        PCs = ['abs_PC{}_zscore'.format(n+1) for n in range(n_components)]

    y = 'momenta_norm_zscore'
    X = ' + '.join(PCs)
    pfile = outfile.replace('.xlsx', '_SLR{}.png'.format('_abs' if abs else ''))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    colors = ['b', 'r', 'g', 'm', 'y']
    markers = ['o', '^', 's', 'd', 'P']

    # only European vs. African
    anova_table, slr = sutils.my_anova_linear(df, X, y, pfile, ax=ax[0], group_names='Group_names', colors=colors, markers=markers, showit=showit)

    cutils.df_save_to_excel(outfile, anova_table, 'LinReg_ANOVA{}_ncomp{}'.format('_abs' if abs else '', n_components))
    cutils.df_save_to_excel(outfile, slr, 'LinReg_SLR{}_ncomp{}'.format('_abs' if abs else '', n_components))

    # European vs. African, Male vs. Female
    colors = ['b', 'deepskyblue', 'k', 'r', 'orange', 'k', 'g', 'm', 'y']
    markers = ['o', 'o', 'o', '^', '^', '^', 's', 'd', 'P']
    anova_table, slr = sutils.my_anova_linear(df, X, y, pfile, ax=ax[1], group_names='Group_names_2', colors=colors, markers=markers, showit=showit)


def plot_embeddings_2d(df, subset1, subset2, pfile, plot_pcs=[1,2], ids=False, showit=True, axis_label=None):

    PCs = ['PC{}'.format(n) for n in plot_pcs]
    X = df[PCs].to_numpy()

    spec=None
    if ids:
        spec = [x[x.find('_')+1:] for x in df['specimen']] 
    
    colors = ['b', 'deepskyblue', 'k', 'r', 'orange', 'k', 'g', 'm', 'y']
    colors2 = np.array([0, 1, 5, 4, 3, 5, 3, .8, 3])
    markers = ['o', 'o', 'o', '^', '^', '^', 's', 'd', 'P']
    
    extend = np.asarray([abs(np.max(X[:,0])) + abs(np.min(X[:,0])), abs(np.max(X[:,1])) + abs(np.min(X[:,1]))])
    border = extend*20/100
    limits = [np.min(X[:,0])-border[0]/2, np.max(X[:,0])+border[0]/2, np.min(X[:,1])-border[1]/2, np.max(X[:,1])+border[1]/2]
    # limits = [np.min(X[:,0])-0.1, np.max(X[:,0])+0.1, np.min(X[:,1])-0.1, np.max(X[:,1])+0.1]

    # plot embeddings in 2D
    fig, ax = plt.subplots(1, 4, figsize=(25, 5))

    # Taxonomy
    pltutils.scatter_classes(X,df['Group_ids'],np.asarray([]),subset1,'Taxonomy',ax=ax[0], ids=spec, limits=limits,img=pfile, showit=False, axis_label=axis_label)
    # Taxonomy + Gender
    pltutils.scatter_classes(X,df['Group_ids_2'],np.asarray([]),subset2,'Taxonomy+Gender',ax=ax[1], limits=limits,ids=spec,img=pfile, colors=colors, markers=markers, colors2=colors2, showit=False, axis_label=axis_label)

    # Subgroups African/European + Gender
    colors2 = np.array([0, 1, 5, 5, 4, 5, 3, .8, 3])
    ns = 3
    for si, sub in enumerate(subset1[:2]):

        g2 = list(subset2[si*ns:si*ns+ns])

        # print(g2)

        col = colors[si*ns:si*ns+ns]
        col2 = list(colors2[si*ns:si*ns+ns])
        mk = markers[si*ns:si*ns+ns]
        if len(subset1)>2:
            dn = 2 if not 'Early_Homo?' in subset1 else 3
            g2 += list(subset1[-dn:])
            col += colors[-3:]
            col2 += list(colors2[-3:])
            mk += markers[-3:]


        df_sub = df.loc[df['Group_names_2'].isin(g2)]

        g2_set = list(set(df_sub['Group_ids_2']))
        g2_ids = np.array([g2_set.index(s) for s in df_sub['Group_ids_2'] ])

        g2_spec = None
        if ids:
            g2_spec = [x[x.find('_')+1:] for x in df_sub['specimen']] 

        pltutils.scatter_classes(df_sub[PCs].to_numpy(),g2_ids,np.asarray([]),g2,'Taxonomy+Gender',ax=ax[si+2],limits=limits, ids=g2_spec,img=pfile, colors=col, markers=mk, colors2=col2, showit=False, axis_label=axis_label)

    if showit:
        plt.show()

    # Correlation with variables
    fig, ax = plt.subplots(1, 2, figsize=(13.4, 5))

    pfile = pfile.replace('.png', '_fun.png')
    # pfile=None
    colors2 = np.array([.5, 6, 3, .8, 4, 2, 1])
    age = np.divide(df['age_in_days'].tolist(), 365)

    levels = np.asarray([0,1,2,3,5,6,7,10,13,16])
    cmap_age = pltutils.nlcmap(plt.cm.jet, levels)

    pltutils.scatter_classes(X,df['Group_ids'],age,subset1,'Age in years',ax=ax[0], fig=fig, ids=spec, limits=limits,img=pfile, colors2=colors2, showit=False, axis_label=axis_label, cmap=cmap_age)
    mnorm = df['momenta_norm'].to_numpy()
    pltutils.scatter_classes(X,df['Group_ids'],mnorm,subset1,'Momenta norm',ax=ax[1], ids=spec, limits=limits,img=pfile, colors2=colors2, showit=False, axis_label=axis_label, cmap=None)

    if showit:
        plt.show()


def plot_embeddings_3d(df, subset1, subset2, pfile, gifs, plot_pcs=[1,2,3], showit=True, axis_label=None):

    PCs = ['PC{}'.format(n) for n in plot_pcs]
    X = df[PCs].to_numpy()

    # fig = plt.figure(figsize=(10, 4))
    # # ax1 = fig.add_subplot(121, projection='3d')
    # # ax2 = fig.add_subplot(122, projection='3d')
    # fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    
    # colors = ['b', 'deepskyblue', 'k', 'r', 'orange', 'k', 'g', 'm']
    # markers = ['o', 'o', 'o', '^', '^', '^', 's', 'd']
    # pltutils.scatter_classes3(X,df['Group_ids'],subset1, ax=None,ids=None,title='Taxonomy', pfile=pfile)
    # pltutils.scatter_classes3(X,df['Group_ids_2'],subset2, ax=None,ids=None,title='Taxonomy', colors=colors, markers=markers, pfile=pfile.replace('.png', '1.png'), showit=False)

    # rotations
    colors = ['b', 'deepskyblue', 'k', 'r', 'orange', 'k', 'g', 'm', 'y']
    colors2 = np.array([0, 1, 5, 4, 3, 5, 3, .8, 3])
    markers = ['o', 'o', 'o', '^', '^', '^', 's', 'd', 'P']

    colors = ['b', 'deepskyblue', 'k', 'r', 'orange', 'k', 'g', 'm', 'y']
    colors2 = np.array([0, 1, 5, 4, 3, 5, 3, .8, 3])
    markers = ['o', 'o', 'o', '^', '^', '^', 's', 'd', 'P']

    if axis_label is None:
        axis_label = ['PC1', 'PC2', 'PC3']
    
    #
    #   Visualization: Europeans vs. Africans
    #
    animator = pltutils.ScatterClasses3Animation(X=df[PCs].to_numpy(), X_classes=df['Group_ids'], X_fun=np.asarray([]), classes=subset1,
                                                ids=None, title='Taxonomy', axis_label=axis_label)
    animator.create_animation(gifs[0], fps=10, view=np.linspace(0, 360, 120))

    
    #
    #   Visualization: Europeans (Male and Female) vs. Africans (Male and Female)
    #
    animator2 = pltutils.ScatterClasses3Animation(X=df[PCs].to_numpy(), X_classes=df['Group_ids_2'], X_fun=np.asarray([]), classes=subset2,
                                                ids=None, title='Taxonomy', axis_label=axis_label, colors=colors, markers=markers)
    animator2.create_animation(gifs[1], fps=10, view=np.linspace(0, 360, 120))
    
    #
    #   Visualization: Europeans (Male vs. Female) and Africans (Male vs. Female)
    #
    ns = 3
    for si, sub in enumerate(subset1[:2]):
        g2 = list(subset2[si*ns:si*ns+ns])
        col = colors[si*ns:si*ns+ns]
        col2 = list(colors2[si*ns:si*ns+ns])
        mk = markers[si*ns:si*ns+ns]
        if len(subset1)>2:
            g2 += list(subset1[-2:])
            col += colors[-2:]
            col2 += list(colors2[-2:])
            mk += markers[-2:]

        df_sub = df.loc[df['Group_names_2'].isin(g2)]

        g2_set = list(set(df_sub['Group_ids_2']))
        g2_ids = np.array([g2_set.index(s) for s in df_sub['Group_ids_2'] ])

        animatorS = pltutils.ScatterClasses3Animation(X=df_sub[PCs].to_numpy(), X_classes=g2_ids, X_fun=np.asarray([]), classes=g2,
                                                ids=None, title='Taxonomy', axis_label=axis_label, colors=col, markers=mk)
        animatorS.create_animation(gifs[si+2], fps=10, view=np.linspace(0, 360, 120))
