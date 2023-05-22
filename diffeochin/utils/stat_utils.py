## for data
import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for statistical tests
import scipy
import scipy.stats as stats
import statsmodels.api as sm
from scipy.special import expit

import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from sklearn.model_selection import cross_val_predict, LeaveOneOut, permutation_test_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing as preprocessing
from sklearn.svm import SVR

from . import class_utils as cutils


# def within_group_variation()
# '''
# Within group variation:
# How much do the individuals vary from their group mean.
# '''

# '''
# Between group variation:
# How much do the group means vary from the overall mean.
# '''

'''
Plot distributions and outliers of numerical variable
:parameter
    :param df: dataframe - input data
    :param col: str - name of the column to analyze
'''
def plot_numerical(df, col, fig, ax0, ax1):
    # fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False)
    # fig.suptitle(col, fontsize=20)### distribution
    ax0.title.set_text('distribution')
    variable = df[col].fillna(df[col].mean())
    breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
    variable = variable[ (variable > breaks[0]) & (variable < 
                        breaks[10]) ]
    sns.distplot(variable, hist=True, kde=True, kde_kws={"shade": True}, ax=ax0)
    des = df[col].describe()
    ax0.axvline(des["25%"], ls='--')
    ax0.axvline(des["mean"], ls='--')
    ax0.axvline(des["75%"], ls='--')
    ax0.grid(True)
    des = round(des, 2).apply(lambda x: str(x))
    box = '\n'.join(("min: "+des["min"], "25%: "+des["25%"], "mean: "+des["mean"], "75%: "+des["75%"], "max: "+des["max"]))
    ax0.text(0.95, 0.95, box, transform=ax0.transAxes, fontsize=10, va='top', ha="right", bbox=dict(boxstyle='round', facecolor='white', alpha=1))### boxplot 
    # ax1.title.set_text('outliers (log scale)')
    # tmp_dtf = pd.DataFrame(df[col])
    # tmp_dtf[col] = np.log(tmp_dtf[col])
    # # tmp_dtf[col] = tmp_dtf[x]
    # tmp_dtf.boxplot(column=col, ax=ax1)
    # # plt.show()


"""
test for Pearson's correlation coefficient to compare two numerical variables
:parameter
    :param df: dataframe - input data
    :param num1: str - name of the first numerical column to analyze
    :param num2: str - name of the second numerical column to analyze
"""
def myPearsonTest(num1, num2, df):
    # dtf_noNan = df[df[num1].notnull()]
    dtf_noNan = df.dropna(subset=[num1, num2])
    coeff, p = scipy.stats.pearsonr(dtf_noNan[num1], dtf_noNan[num2])
    coeff, p = round(coeff, 3), round(p, 3)
    conclusion = "Significant" if p < 0.05 else "Non-Significant"
    print("Pearson Correlation:", coeff, conclusion, "(p-value: "+str(p)+")")

    return coeff, conclusion, p


'''
Plot distributions and outliers of numerical variable
:parameter
    :param df: dataframe - input data
    :param col: str - name of the column to analyze
'''
def plot_numerical(df, col, fig, ax0, ax1):
    # fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False)
    # fig.suptitle(col, fontsize=20)### distribution
    ax0.title.set_text('distribution')
    variable = df[col].fillna(df[col].mean())
    breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
    variable = variable[ (variable > breaks[0]) & (variable < 
                        breaks[10]) ]
    sns.distplot(variable, hist=True, kde=True, kde_kws={"shade": True}, ax=ax0)
    des = df[col].describe()
    ax0.axvline(des["25%"], ls='--')
    ax0.axvline(des["mean"], ls='--')
    ax0.axvline(des["75%"], ls='--')
    ax0.grid(True)
    des = round(des, 2).apply(lambda x: str(x))
    box = '\n'.join(("min: "+des["min"], "25%: "+des["25%"], "mean: "+des["mean"], "75%: "+des["75%"], "max: "+des["max"]))
    ax0.text(0.95, 0.95, box, transform=ax0.transAxes, fontsize=10, va='top', ha="right", bbox=dict(boxstyle='round', facecolor='white', alpha=1))### boxplot 
    # ax1.title.set_text('outliers (log scale)')
    # tmp_dtf = pd.DataFrame(df[col])
    # tmp_dtf[col] = np.log(tmp_dtf[col])
    # # tmp_dtf[col] = tmp_dtf[x]
    # tmp_dtf.boxplot(column=col, ax=ax1)
    # # plt.show()
    
'''
Plot distributions and outliers of categorical variable
:parameter
    :param df: dataframe - input data
    :param col: str - name of the column to analyze
'''
def plot_categorical(df, col, ax):
    # ax = df[col].value_counts().sort_values().plot(kind="barh")
    categories = df[col].unique()[~np.isnan(df[col].unique())]
    ax.barh(categories,df[col].value_counts().to_numpy())
    # ax.barh(df[col].value_counts().sort_values().to_numpy(), [0,1])
    totals= []
    for i in ax.patches:
        totals.append(i.get_width())
    total = sum(totals)
    for i in ax.patches:
        ax.text(i.get_width()+.3, i.get_y()+.20, 
        str(round((i.get_width()/total)*100, 2))+'%', 
        fontsize=10, color='black')
    ax.set_yticks(categories)
    ax.grid(axis="x")
    # plt.suptitle(col, fontsize=20)
    # plt.show()


def summary_to_df(model):
    # Note that tables is a list. The table at index 1 is the "core" table. Additionally, read_html puts dfs in a list, so we want index 0

    results_as_html = model.summary().tables[1].as_html()
    df = pd.read_html(results_as_html, header=0, index_col=0)[0]

    return df


def plot_linear_regression(df, variables, pred, slope, intercept, ax=None, title=None, group_names='Group_names', group_ids='Group_ids', colors=None, markers=None, pfile=None, showit=True):

    x = df[variables[0]]
    y = df[variables[1]]

    if ax is None:
        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=[5,4])
        ax.grid(True)
    # ax.scatter(x, y, color='black')

    used = set()
    groups = [x for x in df[group_names] if x not in used and (used.add(x) or True)]
    
    if colors is None:
        colors = ['b', 'r', 'g', 'm', 'y', 'b', 'k']
    if markers is None:
        markers = ['o', '^', 's', 'd', '^', 'X', 'P']

    for gi, g in enumerate(groups):
        df_sub = df.loc[df[group_names] == g]
        xg = df_sub[variables[0]]
        yg = df_sub[variables[1]]
        ax.plot(xg, yg, markers[gi], color=colors[gi], alpha=.5, label=g)

    ax.set_xlabel(variables[0])
    ax.set_ylabel(variables[1])

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    n = x.size                        # number of samples
    m = 2                             # number of parameters
    dof = n - m                       # degrees of freedom
    t = stats.t.ppf(0.975, dof)       # Students statistic of interval confidence
    
    residual = y - pred 
    std_error = (np.sum(residual**2) / dof)**.5   # Standard deviation of the error
    # calculating the r2
    # https://www.statisticshowto.com/probability-and-statistics/coefficient-of-determination-r-squared/
    # Pearson's correlation coefficient
    numerator = np.sum((x - x_mean)*(y - y_mean))
    denominator = ( np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2) )**.5
    correlation_coef = numerator / denominator
    r2 = correlation_coef**2

    # mean squared error
    MSE = 1/n * np.sum( (y - pred)**2 )

    # to plot the adjusted model
    x_line = np.linspace(np.min(x), np.max(x), 100)
    if slope is not None and intercept is not None:
        y_line = np.polyval([slope, intercept], x_line)
    else:
        y_line = pred

    # confidence interval
    ci = t * std_error * (1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))**.5
    # predicting interval
    pi = t * std_error * (1 + 1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))**.5  

    
    ax.plot(x_line, y_line, color='gray', linewidth=2)
    # ax.fill_between(x_line, y_line + pi, y_line - pi, color = 'lightcyan', label = '95% prediction interval')
    ax.fill_between(x_line, y_line + ci, y_line - ci, color = 'gainsboro', label = '95% confidence interval')

    # rounding and position must be changed for each case and preference
    a = str(np.round(intercept))
    b = str(np.round(slope,3))
    r2s = str(np.round(r2,3))
    MSEs = str(np.round(MSE))

    # ax.text(-1, 3, 'y = ' + a + ' + ' + b + ' x')
    # ax.text(-1, 2.5, '$r^2$ = ' + r2s + '     MSE = ' + MSEs)

    ax.legend(loc='best', fontsize=8)
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('$r^2$ = ' + r2s + '; MSE = ' + MSEs)

    if pfile is not None:
        plt.savefig(pfile, dpi=300)

    if showit:
        plt.show()

def plot_svr_regression(df, variables, x, y, pred, r2, ax=None, title=None, group_names='Group_names', group_ids='Group_ids', colors=None, markers=None, pfile=None, showit=True):


    if ax is None:
        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=[5,4])
        ax.grid(True)
    # ax.scatter(x, y, color='black')

    used = set()
    groups = [x for x in df[group_names] if x not in used and (used.add(x) or True)]
    
    if colors is None:
        colors = ['b', 'r', 'g', 'm', 'y', 'b', 'k']
    if markers is None:
        markers = ['o', '^', 's', 'd', '^', 'X', 'P']

    for gi, g in enumerate(groups):
        df_sub = df.loc[df[group_names] == g]
        xg = df_sub[variables[0]]
        yg = df_sub[variables[1]]
        ax.plot(xg, yg, markers[gi], color=colors[gi], alpha=.5, label=g)

    ax.set_xlabel(variables[0])
    ax.set_ylabel(variables[1])

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    n = x.size                        # number of samples
    m = 2                             # number of parameters
    dof = n - m                       # degrees of freedom
    t = stats.t.ppf(0.975, dof)       # Students statistic of interval confidence
    
    residual = y - pred 
    std_error = (np.sum(residual**2) / dof)**.5   # Standard deviation of the error

    # mean squared error
    MSE = 1/n * np.sum( (y - pred)**2 )

    # confidence interval
    ci = t * std_error * (1/n + (x[:,0] - x_mean)**2 / np.sum((x - x_mean)**2))**.5
    # predicting interval
    pi = t * std_error * (1 + 1/n + (x[:,0] - x_mean)**2 / np.sum((x - x_mean)**2))**.5  

    
    ax.plot(x, pred, color='gray', linewidth=2)
    # # ax.fill_between(x_line, y_line + pi, y_line - pi, color = 'lightcyan', label = '95% prediction interval')
    ax.fill_between(x[:,0], pred + ci, pred - ci, color = 'gainsboro', label = '95% confidence interval')

    # # rounding and position must be changed for each case and preference
    r2s = str(np.round(r2,3))
    MSEs = str(np.round(MSE))

    ax.legend(loc='best', fontsize=8)
    if title is not None:
        ax.set_title(f'{title}: $r^2$ = {r2s}; MSE = {MSEs} ')
    else:
        ax.set_title('$r^2$ = ' + r2s + '; MSE = ' + MSEs)

    if pfile is not None:
        plt.savefig(pfile, dpi=300)

    if showit:
        plt.show()


def plot_logistic_regression(df, variables, coef, intercept, title=None, ax=None, group_names='Group_names', group_ids='Group_ids', colors=None, markers=None, pfile=None, showit=True):

    x = df[variables[0]]
    y = df[variables[1]]

    if ax is None:
        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=[5,4])
    ax.grid(True)
    # ax.scatter(x, y, color='black')

    used = set()
    groups = [x for x in df[group_names] if x not in used and (used.add(x) or True)]
    
    if colors is None:
        colors = ['b', 'r', 'g', 'm', 'y', 'b', 'k']
    if markers is None:
        markers = ['o', '^', 's', 'd', '^', 'X', 'P']

    for gi, g in enumerate(groups):
        df_sub = df.loc[df[group_names] == g]
        # print(df_sub.shape)
        xg = df_sub[variables[0]]
        yg = df_sub[variables[1]]
        ax.plot(xg, yg, markers[gi], color=colors[gi], alpha=.5, label=g)

    ax.set_xlabel(variables[0])
    ax.set_ylabel(variables[1])

    # to plot the adjusted model
    x_line = np.linspace(np.min(x), np.max(x), 100)
    y_line = expit(x_line * coef + intercept).ravel()

    ax.plot(x_line, y_line, color='gray', linewidth=2)


    ax.legend(loc='best', fontsize=8)
    if title is not None:
        ax.set_title(title)

    if pfile is not None:
        plt.savefig(pfile, dpi=300)

    if showit:
        plt.show()



"""
one-way ANOVA test to compare a categorical and a numerical variables
:parameter
"""
def myANOVA(model):
    table = sm.stats.anova_lm(model)
    p = table["PR(>F)"][0]
    coeff, p = None, round(p, 3)
    conclusion = "Correlated" if p < 0.05 else "Non-Correlated"
    print("Anova F: the variables are", conclusion, "(p-value: "+str(p)+")")



def my_logistic_regression(X, y, n_permutations=100, n_jobs=10):

    '''
        LeaveOneOut Cross-validation
    '''
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    cv = LeaveOneOut()
    preds = cross_val_predict(model, X, y, cv=cv, method='predict_proba')

    '''
        Significance testing
    '''
    score, _, pvalue = permutation_test_score(model, X, y, scoring="balanced_accuracy", cv=cv, n_permutations=n_permutations, n_jobs=n_jobs)
    ptest = [score, pvalue]

    return preds, ptest


def my_anova_logistic(df, X, y, pfile, ax=None, group_names='Group_names', colors=None, markers=None, showit=True):

    if not np.all(df[y]<2):
        print('conversion')
        df[y] = np.where(np.array(df[y])==3, 0, 1)

    myformula = '{} ~ {}'.format(y, X)
    model = smf.ols(myformula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # cutils.df_save_to_excel(outfile.replace('.xlsx', '_taxon.xlsx'), anova_table, sheet_name)

    # find most significant PC for age regression
    pc = np.argmin(anova_table['PR(>F)'])

    # simple logistic regression
    X = X.split(' + ')[pc]
    myformula = '{} ~ {}'.format(y, X)
    print(myformula)
    model = smf.logit(y+' ~ '+X, data=df).fit()
    predictions = model.predict(df[X])
    df_res = summary_to_df(model)

    # title = '{}_PC{}'.format(sheet_name,pc)
    # colors = None
    # markers = None
    # pfile = '{}{}.png'.format(pfile, pc+1) 
    plot_logistic_regression(df, [X, y], model.params[X], model.params.Intercept, ax=ax, group_names=group_names, group_ids=y, colors=colors, markers=markers, pfile=pfile, showit=showit)
            
    return anova_table, df_res


def my_anova_linear(df, X, y, pfile, ax, group_names='Group_names', colors=None, markers=None, showit=True):

    Xa = X

    myformula = '{} ~ {}'.format(y, X)
    print(myformula)
    model = ols(myformula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # cutils.df_save_to_excel(outfile.replace('.xlsx', '_taxon.xlsx'), anova_table, sheet_name)

    # find most significant PC for age regression
    pc = np.argmin(anova_table['PR(>F)'])

    # # simple linear regression
    
    X = X.split(' + ')[pc]
    myformula = '{} ~ {}'.format(y, X)
    print(myformula)
    model = ols(y+' ~ '+X, data=df).fit()
    predictions = model.predict(df[X])
    df_res = summary_to_df(model)

    # title = '{}_PC{}'.format(sheet_name,pc)
    # colors = None
    # markers = None
    # pfile = '{}{}.png'.format(pfile, pc+1) 
    plot_linear_regression(df, [X, y], predictions, model.params[X], model.params.Intercept, ax=ax, group_names=group_names, group_ids=y, colors=colors, markers=markers, pfile=pfile, showit=showit)
            
    coeff, concl, pval = myPearsonTest(X, y, df)

    df_res["Pearson's (r)"] = coeff
    df_res["Pearson's (p-val)"] = pval
    df_res['R squared'] = coeff**2

    return anova_table, df_res, pc

def my_nonlinear_regression(df, X, y, pc, pfile, ax, group_names='Group_names', colors=None, markers=None, showit=True):
    X = X.split(' + ')#[pc]

    """
        Non-linear regression
    """
    # variables = [X.split(' + '), y]
    variables = [X[pc], y]
    df2 = df.dropna(subset=[y])
    # X = df2[X.split(' + ')].to_numpy()
    X = df2[X].to_numpy()
    y = df2[y].to_numpy()

    

    svr_rbf = SVR(kernel="rbf", C=100, gamma='auto', epsilon=1)
    svr_lin = SVR(kernel="linear", C=100, gamma="auto")
    svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)
    kernel_label = ["RBF", "Polynomial", "Linear"]

    print()
    print('Regression with SVR')
    print('All PCs')

    # r2 for regression with all PCs
    svrs = [svr_rbf, svr_poly, svr_lin]
    r2_all = []
    for ix, svr in enumerate(svrs):

        svr.fit(X, y)
        r2_all.append(svr.score(X,y))
        print(f'{kernel_label[ix]}: R2 = {r2_all[ix]:1.3f}')

    # svrs = [svr_rbf, svr_poly]
    # kernel_label = ["RBF","Polynomial"]

    sort = np.argsort(X[:,pc])
    y = y[sort]
    X = np.expand_dims(X[sort,pc], axis=1)
    
    r2 = []
    print('Significant PC')
    for ix, svr in enumerate(svrs):

        svr.fit(X, y)
        r2.append(svr.score(X,y))
        y_pred = svr.predict(X)
        print(f'{kernel_label[ix]}: R2 = {r2[ix]:1.3f}')

        if not kernel_label[ix]=='Linear':

            plot_svr_regression(df, variables, X, y, y_pred, ax=ax[ix], r2=r2[ix], title=kernel_label[ix], group_names=group_names, group_ids=y, colors=colors, markers=markers, pfile=pfile, showit=showit)


    return r2, r2_all



def group_variations(df, variables, subgroup):

    """
    within-group-variation: variance of each group individually
    between-group-variation: variation of group means to sample mean
    """

    # only consider such subgroups
    pop_mean = df.mean()
    group_means = df.groupby([subgroup]).mean() # mean of each group
    group_var = df.groupby([subgroup]).var()    # variance of each group
    group_size = df.groupby([subgroup]).size()  # size of each group
    diff = group_means.sub(pop_mean)**2         # (mean_group - sample_mean)^2

    if not isinstance(variables,list):
        within_group_variation = np.multiply((group_size-1).to_numpy(), group_var[variables].to_numpy())
        between_group_variation = np.multiply(group_size.to_numpy(), diff[variables].to_numpy())

    else:    
        within_group_variation = 0
        between_group_variation = 0
        for v in range(len(variables)):
            within_group_variation += np.multiply((group_size-1).to_numpy(), group_var[variables[v]].to_numpy())
            between_group_variation += np.multiply(group_size.to_numpy(), diff[variables[v]].to_numpy())

    index = group_means.index
    groups = list(index)

    df2 = pd.DataFrame

    # create dataframe
    d = {'groups': groups}
    df2 = pd.DataFrame(data=d)
    df2['Within_group_variation'] = within_group_variation
    df2['Between_group_variation'] = between_group_variation

    # statistical test for within-group variation

    # two samples: group 1 and 2 with |s_i-s_mean|
    # unpaired t-test

    # print(group_means[variables])
    group_means = df.groupby([subgroup], as_index=False).mean()
    # print(group_means[variables])
    # print(group_means[group_means[subgroup]=='European'][variables])

    groups = group_means[subgroup].unique()
    # print(groups)
    samples = []
    # group_means_np = group_means[variables].to_numpy()
    for gi, g in enumerate(groups):
        if g=='European' or g=='African':
            # group_mean = group_means_np[gi,:]
            group_mean = group_means[group_means[subgroup]==g][variables].to_numpy()
            # print(group_mean)
            group_samples = df[df[subgroup]==g][variables].to_numpy()

            # distance to each sample to the group mean
            samples.append(np.linalg.norm(group_samples - group_mean, axis=1))
            # print()
        
    # Means of the two samples
    mean_0 = samples[0].mean()
    mean_1 = samples[1].mean()
    difference = mean_0 - mean_1

    print(f'Difference of means = {difference:5.3f}')

    n_0 = samples[0].shape[0]
    n_1 = samples[1].shape[0]

    print(f'n = {n_0}, n = {n_1}')

    # Sample standard deviations
    std_0 = samples[0].std(ddof=1)
    std_1 = samples[1].std(ddof=1)

    print(f'Standard deviations: {std_0:5.3f} and {std_1:5.3f}')

    # Standard errors of the means
    sem_0 = std_0 / np.sqrt(n_0)
    sem_1 = std_1 / np.sqrt(n_1)

    print(f'Standard errors of the means: {sem_0:5.3f} and {sem_1:5.3f}')

    # Standard error of the difference
    sed = np.sqrt(std_0**2 / n_0 + std_1**2 / n_1)

    print(f'Standard error of the difference: {sed:5.3f}')

    # Confidence intervals
    upper_ci = difference + 2 * sed
    lower_ci = difference - 2 * sed

    print(f'Difference of the means = {difference:5.3f} ({lower_ci:5.3f} to {upper_ci:5.3f})')

    # Two-sample t-test
    statistic, pvalue = stats.ttest_ind(samples[0], samples[1])

    print(f'Two-sample t-test: s = {statistic:5.3f}, p = {pvalue:5.3f}')

    statistics = len(groups)*[0]
    statistics[list(groups).index('European')] = pvalue
    statistics[list(groups).index('African')] = pvalue

    df2['Unpaired t-test (p-value)'] = statistics

    return df2