'''
This script does post-analysis with classification results
Input: gas/bulk_test_pred.csv
Implemented functions are:
1. extract misclassified results (predicted class and probs) as dataframe
2. individual scatter plot
3. UMAP
'''
import _io
import logging

import pandas as pd
import numpy as np
import pathlib, time, sys, re, datetime
from pathlib import Path, WindowsPath
import itertools
import matplotlib.pyplot as plt
import seaborn as sb
import umap
import umap.plot as umplt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pickle


def extract_misclassify(in_p: pathlib.Path, logger: logging.getLogger()) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    '''
    Extract misclassfied dataframe
    Parameters:
        in_p - the path of prediction result file, having original indexes, IDs, labels, predicted classes, prediction probs for each class
    '''
    df = pd.read_csv(in_p, index_col=0)
    #print(df.columns, file=log_outf)
    logger.info(df.columns)
    misdf = df[df['Label'] != df['Pred']]
    #print('There are {} samples, {} are wrongly classified\n'.format(df.shape[0], misdf.shape[0]), file=log_outf)
    #print(misdf[['rtu-ioa', 'Label', 'Pred']].value_counts(), file=log_outf)
    logger.info('There are {} samples, {} are wrongly classified\n'.format(df.shape[0], misdf.shape[0]))
    mis_count_df = misdf[['rtu-ioa', 'Label', 'Pred']].value_counts().reset_index().rename(columns={0: 'Count'})
    mis_count_df.sort_values(by=['Label', 'Pred', 'Count'], ascending=False, inplace=True)
    logger.info(mis_count_df)
    return df, misdf, mis_count_df


def scatter_one(x: np.ndarray, y: np.ndarray, out_f: pathlib.Path):
    '''
    Plot individual scatter plot
    '''
    plt.figure()
    plt.scatter(x, y)
    plt.savefig(out_f)


def manifold_umap(point_var: str, point_x: np.ndarray, label_class: str, label_x: np.ndarray, pred_class: str,
                  pred_x: np.ndarray, out_p: pathlib.Path, logger: logging.getLogger()):
    """
    # TODO: 4 plots of manifold learning: correct predictions for both classes in uni-color,
    # wrong prediction + correct class 1 and wrong + correct class 2 in different colors

    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 12))
    # 3 UMAP learning for all label class, all pred class, for
    umap_params = {
        'n_neighbors': 20,
        'min_dist': 0.8,
        'n_component': 2,
        'random_state': 2022
    }
    #n_neighbors_list = [5, 10, 15, 20, 25, 30, 50, 70, 100]
    #n_neighbors_list = [15, 50, 100, 500, 1000]
    #min_dist_list = [0.2, 0.5, 0.8]
    n_neighbors_list = [20]
    min_dist_list = [0.5]
    for (nn, md) in list(itertools.product(n_neighbors_list, min_dist_list)):
    # TODO: think about umap separately learn point and correct classes or together
        #print('Start UMAP, please be patient...')
        logger.info('Start UMAP, please be patient...')
        start_t = time.time()
        label_embedding = umap.UMAP(n_neighbors=nn, min_dist=md, n_components=umap_params['n_component'],
                                    random_state=umap_params['random_state']).fit_transform(np.vstack((point_x, label_x)))
        pred_embedding = umap.UMAP(n_neighbors=nn, min_dist=md, n_components=umap_params['n_component'],
                                    random_state=umap_params['random_state']).fit_transform(np.vstack((point_x, pred_x)))
        point_embedding = umap.UMAP(n_neighbors=nn, min_dist=md, n_components=umap_params['n_component'],
                                    random_state=umap_params['random_state']).fit_transform(point_x)
        #label_embedding = umap.UMAP(n_neighbors=nn, min_dist=md, n_components=umap_params['n_component'],
         #                           random_state=umap_params['random_state']).fit_transform(label_x)
        #pred_embedding = umap.UMAP(n_neighbors=nn, min_dist=md, n_components=umap_params['n_component'],
         #                          random_state=umap_params['random_state']).fit_transform(pred_x)

        #point_embedding = umap.UMAP(n_neighbors=umap_params['n_neighbors'], min_dist=umap_params['min_dist'], random_state=umap_params['random_state']).fit_transform(point_x)
        #label_embedding = umap.UMAP(n_neighbors=umap_params['n_neighbors'], min_dist=umap_params['min_dist'], random_state=umap_params['random_state']).fit_transform(label_x)
        #pred_embedding = umap.UMAP(n_neighbors=umap_params['n_neighbors'], min_dist=umap_params['min_dist'], random_state=umap_params['random_state']).fit_transform(pred_x)
        #print('**** UMAP takes {} minutes ****'.format((time.time() - start_t) / 60), file=log_outf)
        logger.info('**** UMAP takes {} minutes ****'.format((time.time() - start_t) / 60))
        fig, axes = plt.subplots(1, 2, figsize=(20, 12))
        #axes[0].scatter(label_embedding[:, 0], label_embedding[:, 1], s=16, c='#488FB1')
        #axes[0].scatter(point_embedding[:, 0], point_embedding[:, 1], s=16, c='#F68989')
        axes[0].scatter(label_embedding[point_x.shape[0]:, 1], label_embedding[point_x.shape[0]:, 1], s=16, c='#488FB1')
        axes[0].scatter(label_embedding[:point_x.shape[0], 1], label_embedding[:point_x.shape[0], 1], s=16, c='#F68989')

        axes[0].set_title('Label: {}'.format(label_class))
        #axes[1].scatter(pred_embedding[:, 0], pred_embedding[:, 1], s=16, c='#488FB1')
        #axes[1].scatter(point_embedding[:, 0], point_embedding[:, 1], s=16, c='#F68989')
        axes[1].set_title('Predicted: {}'.format(pred_class))
        #axes[2].scatter(point_embedding[:, 0], point_embedding[:, 1], point_embedding[:, 2], c='#488FB1')
        #axes[2].set_title('Point {}'.format(point_var))
        fig.suptitle('UMAP Projection: n_neighbors={}, min_dist={}, random_state={}'.format(
            nn, md, umap_params['random_state']))

        out_f = Path(out_p, '{}_mis_umap.svg'.format(point_var))
        plt.title('UMAP Projection: n_neighbors={}, min_dist={}, random_state={}'.format(umap_params['n_neighbors'], umap_params['min_dist'], umap_params['random_state']), fontsize=24)
        plt.savefig(out_f)
        plt.show()
        logger.info('Export UMAP figure to: {}'.format(out_f))


def plot_shap_force(clf, x_test, features):
    """
    SHAP force plot for partial samples
    """
    import shap
    print('****** Start SHAP plotting *******')
    x_test = pd.DataFrame(data=x_test, columns=features)
    # explainer = shap.TreeExplainer(clf)
    explainer = shap.Explainer(clf, output_names=clf.classes_, feature_names=features)
    # visualize partial predictions
    shap_values_test = explainer.shap_values(x_test)
    #shap.force_plot(explainer.expected_value[2], shap_values_test[2][0, :], feature_names=features, matplotlib=True, show=False)
    shap.force_plot(explainer.expected_value[2], shap_values_test[2][757, :], feature_names=features, matplotlib=True, show = False)
    fig, ax = plt.gcf(), plt.gca()
    fig.set_size_inches(30, 10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
    # ax.legend(prop=dict(size=28))
    fig.savefig("shap_force_single.svg", bbox_inches='tight')
    shap.force_plot(explainer.expected_value[2], shap_values_test[2][766, :], feature_names=features, matplotlib=True,
                    show=False)
    fig, ax = plt.gcf(), plt.gca()
    fig.set_size_inches(30, 10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
    # ax.legend(prop=dict(size=28))
    fig.savefig("shap_force_single2.svg", bbox_inches='tight')



def plot_shap_summary(clf, x, x_train, x_test, features, out_subpath, logger):
    """
    SHAP plots
    1. overall summary bar plot for multi-classification
    2. force plot for partial samples
    """
    import shap
    #print('****** Start SHAP plotting *******', file=log_outf)
    logger.info('****** Start SHAP plotting *******')
    x_test = pd.DataFrame(data=x_test, columns=features)
    #explainer = shap.TreeExplainer(clf)
    explainer = shap.Explainer(clf, output_names=clf.classes_, feature_names=features)
    #explainer = shap.Explainer(clf, x_test, output_names=clf.classes_, feature_names=features)
    logger.info('Finish extraction of explainer')
    # visualize the contribution in predictions
    shap_values_test = explainer.shap_values(x_test)
    #shap_values_test = explainer(x_test)
    shap_pkl = open(Path(out_subpath, 'shap_values_test.pkl'), 'wb')
    pickle.dump(shap_values_test, shap_pkl)
    logger.info('Finish applying explainer on test data')
    # summarize the effects of all the features
    # plt.figure(figsize=(12, 40))
    out_figpath = Path(out_subpath, "shap_summary.svg")
    # TODO: Refer to Github, color parameter in shap/plots/colors
    # summary_plot is summary_legacy of _beeswarm plot
    shap.summary_plot(shap_values_test, x_test, class_names=clf.classes_, max_display=10, show=False)  # , matplotlib=True, show=False)
    #shap.plots.bar(shap_values_test, max_display=10)
    fig, ax = plt.gcf(), plt.gca()
    fig.set_size_inches(30, 20)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
    #y_tick_label_list = ['Sum of absolute changes', 'Mean in the quantiles of [0.0, 0.6]', 'Lempel-Ziv complexity (bins=100)', 'Sum of reoccuring values', 'Maximum value', 'Minimum value', 'Mean in the quantiles of [0.0, 0.8]', 'Value at quantile 0.1', 'Value at quantile 0.4', 'ASDU normalized measurement type']
    #ax.set_yticklabels(y_tick_label_list, fontsize=20)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
    #patterns = ["|", "\\", "/", "+", "-", ".", "*", "x", "o", "O"]
    ax.legend(prop=dict(size=28))
    fig.savefig(out_figpath, bbox_inches='tight')
    logger.info('Export SHAP summary plot to: {}'.format(out_figpath))
    #for sh in shap_values_test:
        #print(type(sh), sh.shape, file=log_outf)
        #logger.info('Type of sh: {}, shape = {}'.format(type(sh), sh.shape))
    '''
    shap.force_plot(explainer.expected_value[2], shap_values_test[2][0, :], feature_names=features, matplotlib=True, show=False)
    fig, ax = plt.gcf(), plt.gca()
    fig.set_size_inches(30, 10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
    # ax.legend(prop=dict(size=28))
    fig.savefig("shap_force_single.svg", bbox_inches='tight')
    shap.force_plot(explainer.expected_value[2], shap_values_test[2][0, :], x_test.iloc[0, :], matplotlib=True, show=False)
    fig, ax = plt.gcf(), plt.gca()
    fig.set_size_inches(30, 10)
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
    # ax.legend(prop=dict(size=28))
    fig.savefig("shap_force_single2.svg", bbox_inches='tight')
    '''



    #print(type(explainer.expected_value), explainer.expected_value[0].shape)


    #shap.summary_plot(shap_values_test, x_test, class_names=clf.classes_, max_display=20, plot_type="layered_violin", show=False)
    #fig.set_size_inches(30, 20)
    #fig.savefig("shap_summary2.svg", bbox_inches='tight')

    #plt.tight_layout()
    #plt.figure(figsize=(12, 40))
    #shap.plots.beeswarm(explainer(x_test), order=explainer(x_test).abs.max(0), show=False)
    #plt.savefig('shap_beeswarm.svg', bbox_inches='tight')
    #plt.legend(['alarm_config', 'position', 'pressure', 'flow', 'temperature'])
    #plt.savefig()
    #shap.summary_plot(shap_values_test, x_test, plot_type="bar")

    # dependence plots
    #for name in x_test.columns:
    #    shap.dependence_plot(name, shap_values_test, x_test)
    #return


def single_umap_plot(x: np.ndarray, out_f: pathlib.Path, point_var: str):
    """
    OUTDATED!!!
    UMAP manifold learning
    """
    x_scaled = StandardScaler().fit_transform(x)
    print('Start UMAP, please be patient...')
    start_t = time.time()
    embedding = umap.UMAP(n_neighbors=15, min_dist=0.5, random_state=2022).fit_transform(x_scaled)
    print('**** UMAP takes {} minutes ****'.format((time.time() - start_t) / 60))
    print(embedding.shape)
    plt.figure(figsize=(16, 12))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=6
    )
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection for {}'.format(point_var), fontsize=24)
    plt.savefig(out_f)
    plt.show()
