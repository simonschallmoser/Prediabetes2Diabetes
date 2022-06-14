import numpy as np
import pandas as pd
import matplotlib.font_manager

import matplotlib.pyplot as plt
from shap import plots
import matplotlib.pyplot as pl
from shap.plots._labels import labels
from shap.utils import format_value, ordinal_str
from shap.plots._utils import convert_ordering, convert_color, merge_nodes, get_sort_order, sort_inds, dendrogram_coords
from shap.plots import colors
from shap import Explanation, Cohorts
from sklearn.metrics import roc_auc_score

def get_results(model,
                seeds,
                pred_years,
                return_all_aurocs=False):
    """
    Calculates AUROCs and corresponding ranges over 10 different random seeds.
    """
    if return_all_aurocs:
        pred_years = [pred_years]
    aurocs = []
    lower_range = []
    upper_range = []
    for pred_year in pred_years:
        y_tests = []
        y_preds = []
        for seed in seeds:
            results = np.load(f'results/results_{model}_{pred_year}_{seed}.npy', allow_pickle=True).flatten()[0]
            y_tests.append(results['y_test'])
            y_preds.append(results['y_pred'])
        aurocs_all1 = []
        aurocs_all2 = []
        for i in range(10):
            aurocs_all = []
            for j in range(5):
                aurocs_all.append(roc_auc_score(y_tests[i][j], y_preds[i][j]))
            aurocs_all1.append(np.mean(aurocs_all))
            aurocs_all2.extend(aurocs_all)
        auroc_mean = np.mean(aurocs_all1)
        aurocs.append(auroc_mean)
        lower_range.append(auroc_mean - np.min(aurocs_all1))
        upper_range.append(np.max(aurocs_all1) - auroc_mean)
    
    if return_all_aurocs:
        return aurocs_all2
    else:
        return aurocs, [lower_range, upper_range]
    
def asterisk(p_value):

    if p_value >= 0.05:
        asterisk = 'ns'
    elif p_value < 0.05 and p_value >= 0.01:
        asterisk = '*'
    elif p_value < 0.01 and p_value >= 0.001:
        asterisk = '**'
    elif p_value < 0.001:
        asterisk = '***'

    return asterisk
    
def get_asterisks(model1,
                  model2,
                  pred_years,
                  seeds):
   
    p_values = []
    for pred_year in pred_years:
        p_values.append(mannwhitneyu(get_results(model1, seeds, pred_year, True),
                                     get_results(model2, seeds, pred_year, True)).pvalue)
    asterisk_max = asterisk(np.min(p_values))
    asterisk_min = asterisk(np.max(p_values))
    
    return asterisk_min, asterisk_max

def plot_results(models,
                 labels,
                 seeds,
                 pred_years,
                 file_ending,
                 y_min=0.645,
                 y_max=0.805,
                 x_min=0.75,
                 x_max=7.55,
                 add_significance_levels=True,
                 store=False):
    """
    Plots results of different models (Figure 1A).
    """
    fig = plt.figure(figsize=(12,7), facecolor='white')
    plt.rcParams.update({'axes.facecolor': 'white'})
    
    linestyles = [(0, (5, 10)), 'dashdot', 'dotted', 'dashed', 'solid']
    colors = ['black', '#0F6E3A', '#FF8500', '#CF2F47', '#2F32CF']
    offsets = [0, -0.15, -0.05, 0.05, 0.15]
    if len(models) == 4:
        offsets = [-0.15, -0.05, 0.05, 0.15]
    elif len(models) == 3:
        offsets = [0, -0.15, 0.15]
    elif len(models) == 2:
        offsets = [-0.05, 0.05]
    aurocs_all = []
    for idx, model in enumerate(models):
        aurocs, errors = get_results(model, seeds, pred_years)
        aurocs_all.append(aurocs)
        plt.errorbar(pred_years+offsets[idx], aurocs, yerr=errors, marker='.', linestyle=linestyles[idx],
                     color=colors[idx], markersize=10, capsize=5, label=labels[idx])
    
    if add_significance_levels:
        # Draw grid by hand
        plt.hlines(0.65, x_min, 5, color='grey', linewidth=0.5)
        plt.hlines(0.7, x_min, 5, color='grey', linewidth=0.5)
        plt.hlines(0.75, x_min, 5, color='grey', linewidth=0.5)
        plt.hlines(0.8, x_min, 5, color='grey', linewidth=0.5)
        for pred_year in pred_years:
            plt.vlines(pred_year, y_min, y_max, color='grey', linewidth=0.5)
    else:
        x_max = 5.25
        plt.grid(color='grey', linewidth=0.5)
    
    plt.ylim(y_min, y_max)
    plt.xlim(x_min, x_max) 
    plt.xticks(pred_years, size=16, fontproperties=font)
    plt.yticks([0.65, 0.7, 0.75, 0.8], size=16, fontproperties=font)
    plt.xlabel('Forecast horizon [years]', size=16, fontproperties=font)
    plt.ylabel('AUROC', size=16, fontproperties=font)
       
    plt.legend(fontsize=16, framealpha=1, edgecolor='black', fancybox=False, loc='lower left', prop=font1)
    
    if add_significance_levels:
        offset_x = 5.27
        for model1 in ['logreg_full', 'logreg_simplified', 'fdrsm']:
            for model2 in ['simplified', 'full']:
                y_1 = get_results(model1, seeds, [5])[0][0]
                y_2 = get_results(model2, seeds, [5])[0][0]
                plt.vlines(offset_x, y_1, y_2, color='black')
                plt.hlines(y_1, offset_x-0.05, offset_x+0.0053, color='black')
                plt.hlines(y_2, offset_x-0.05, offset_x+0.0053, color='black')
                asterisk_min, asterisk_max = get_asterisks(model1, model2, pred_years, seeds)
                if asterisk_min == asterisk_max:
                    if asterisk_min == 'ns':
                        fontsize = 12
                        offset_y = 0
                    else:
                        fontsize = 16
                        offset_y = 0.0035
                    plt.text(offset_x+0.02, (y_1 + y_2) / 2 - offset_y, asterisk_min, fontsize=fontsize, fontproperties=font)
                    if len(asterisk_min) == 3:
                        offset_x += 0.25
                    elif len(asterisk_min) == 2:
                        offset_x += 0.2
                    else:
                        offset_x += 0.1
                else:
                    if asterisk_min == 'ns':
                        fontsize_min = 12
                        offset_y_min = 0.0007
                    else:
                        fontsize_min = 16
                        offset_y_min = 0.0035
                    fontsize_max = 16
                    offset_y_max = 0.0035
                    plt.text(offset_x+0.02, (y_1 + y_2) / 2 - offset_y_min,
                             asterisk_min, fontsize=fontsize_min, fontproperties=font)
                    if len(asterisk_min) == 2:
                        offset_x += 0.16
                    elif len(asterisk_min) == 1:
                        offset_x += 0.1
                    #offset_x += len(asterisk_min) * 0.08
                    plt.text(offset_x, (y_1 + y_2) / 2 - 0.0005, '-', fontsize=12, fontproperties=font)
                    offset_x += 0.06
                    plt.text(offset_x, (y_1 + y_2) / 2 - offset_y_max,
                             asterisk_max, fontsize=fontsize_max, fontproperties=font)
                    if len(asterisk_min + asterisk_max) == 5:
                        offset_x += 0.28
                    elif len(asterisk_min + asterisk_max) == 4:
                        offset_x += 0.25
                    elif len(asterisk_min + asterisk_max) == 3:
                        offset_x += 0.15
    
    if store:
        plt.savefig(f'plots/results_{file_ending}.png', dpi=600, bbox_inches='tight')
    
    return 0

def rename_cols(data):
    """
    Renames the predictors for the SHAP summary plot.
    """

    return data.drop('y', axis=1).rename({'test=glucose': 'Glucose', 'test=hba1c(%)': 'HbA1c',
                                                         'test=hdl-cholest.': 'HDL', 'test=sodiumserum': 'Serum sodium',
                                                         'test=creatinineserum': 'Serum creatinine', 
                                                         'bmi': 'BMI', 'icd9_prefix=719_hist': 'ICD-9: 719',
                                                         'test=triglycerides': 'Triglycerides', 'age': 'Age',
                                                         'test=tsh': 'TSH', 'test=bilirubintotal': 'Bilirubin',
                                                         'test=alt(gpt)': 'ALT', 'icd9_prefix=729_hist': 'ICD-9 code: 729',
                                                         'icd9_prefix=786_hist': 'ICD-9 code: 786', 'test=basophils': 'Basophils',
                                                         'test=uricacid': 'Uric acid', 'test=calciumserum': 'Serum calcium',
                                                         'test=monocytes%': 'Monocytes (%)', 
                                                         'icd9_prefix=278_hist': 'ICD-9 code: 278',
                                                         'systolic_bp': 'SBP', 'test=lymphocytes': 'Lymphocytes', 
                                                         '(med_class = 0)': 'Statins', 
                                                         'test=phosphorus': 'Phosphorus'}, axis=1)

labels = {
    'MAIN_EFFECT': "SHAP main effect value for\n%s",
    'INTERACTION_VALUE': "SHAP interaction value",
    'INTERACTION_EFFECT': "SHAP interaction value for\n%s and %s",
    'VALUE': "SHAP value",
    'GLOBAL_VALUE': "mean(|SHAP value|) (average impact on model output magnitude)",
    'VALUE_FOR': "SHAP value for\n%s",
    'PLOT_FOR': "SHAP plot for %s",
    'FEATURE': "Feature %s",
    'FEATURE_VALUE': "Predictor value",
    'FEATURE_VALUE_LOW': "Low",
    'FEATURE_VALUE_HIGH': "High",
    'JOINT_VALUE': "Joint SHAP value",
    'MODEL_OUTPUT': "Model output value"
}

colors = plots.colors

def summary_plot(shap_values, pred_year, features=None, feature_names=None, max_display=10, plot_type=None,
                 color=None, axis_color="#333333", title=None, alpha=1, store=False, show=True, sort=True,
                 color_bar=True, plot_size="auto", layered_violin_max_num_bins=20, class_names=None,
                 class_inds=None,
                 color_bar_label=labels["FEATURE_VALUE"]):
    """
    Generates a SHAP summary plot for each forecast horizon.
    """
    
    x_positions = [[-0.22, -0.331, -1.13], [-0.22, -0.365, -1.245], [-0.22, -0.3315, -1.13],
                   [-0.22, -0.335, -1.135], [-0.22, -0.347, -1.168]]
    x_position = x_positions[pred_year - 1]
    y_position = [-1.6, -2.08, -2.4]
    
    multi_class = False
    if isinstance(shap_values, list):
        multi_class = True
        if plot_type is None:
            plot_type = "bar" # default for multi-output explanations
        assert plot_type == "bar", "Only plot_type = 'bar' is supported for multi-output explanations!"
    else:
        if plot_type is None:
            plot_type = "dot" # default for single output explanations
        assert len(shap_values.shape) != 1, "Summary plots need a matrix of shap_values, not a vector."

    # default color:
    if color is None:
        if plot_type == 'layered_violin':
            color = "coolwarm"
        elif multi_class:
            color = lambda i: colors.red_blue_circle(i/len(shap_values))
        else:
            color = colors.blue_rgb

    # convert from a DataFrame or other types
    if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = features.columns
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif (features is not None) and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    num_features = (shap_values[0].shape[1] if multi_class else shap_values.shape[1])

    if features is not None:
        shape_msg = "The shape of the shap_values matrix does not match the shape of the " \
                    "provided data matrix."
        if num_features - 1 == features.shape[1]:
            assert False, shape_msg + " Perhaps the extra column in the shap_values matrix is the " \
                          "constant offset? Of so just pass shap_values[:,:-1]."
        else:
            assert num_features == features.shape[1], shape_msg

    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(num_features)])

    if use_log_scale:
        pl.xscale('symlog')


    if max_display is None:
        max_display = 20

    if sort:
        # order features by the sum of their effect magnitudes
        if multi_class:
            feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0))
        else:
            feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
        feature_order = feature_order[-min(max_display, len(feature_order)):]
    else:
        feature_order = np.flip(np.arange(min(max_display, num_features)), 0)

    row_height = 0.4
    if plot_size == "auto":
        pl.gcf().set_size_inches(8, len(feature_order) * row_height + 1.5)
    elif type(plot_size) in (list, tuple):
        pl.gcf().set_size_inches(plot_size[0], plot_size[1])
    elif plot_size is not None:
        pl.gcf().set_size_inches(8, len(feature_order) * plot_size + 1.5)
    pl.axvline(x=0, color="#999999", zorder=-1)

    if plot_type == "dot":
        for pos, i in enumerate(feature_order):
            pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
            shaps = shap_values[:, i]
            values = None if features is None else features[:, i]
            inds = np.arange(len(shaps))
            np.random.shuffle(inds)
            if values is not None:
                values = values[inds]
            shaps = shaps[inds]
            colored_feature = True
            try:
                values = np.array(values, dtype=np.float64)  # make sure this can be numeric
            except:
                colored_feature = False
            N = len(shaps)
            # hspacing = (np.max(shaps) - np.min(shaps)) / 200
            # curr_bin = []
            nbins = 100
            quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
            inds = np.argsort(quant + np.random.randn(N) * 1e-6)
            layer = 0
            last_bin = -1
            ys = np.zeros(N)
            for ind in inds:
                if quant[ind] != last_bin:
                    layer = 0
                ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                layer += 1
                last_bin = quant[ind]
            ys *= 0.9 * (row_height / np.max(ys + 1))

            if features is not None and colored_feature:
                # trim the color range, but prevent the color range from collapsing
                vmin = np.nanpercentile(values, 5)
                vmax = np.nanpercentile(values, 95)
                if vmin == vmax:
                    vmin = np.nanpercentile(values, 1)
                    vmax = np.nanpercentile(values, 99)
                    if vmin == vmax:
                        vmin = np.min(values)
                        vmax = np.max(values)
                if vmin > vmax: # fixes rare numerical precision issues
                    vmin = vmax

                assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"

                # plot the nan values in the interaction feature as grey
                nan_mask = np.isnan(values)
                pl.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777", vmin=vmin,
                           vmax=vmax, s=16, alpha=alpha, linewidth=0,
                           zorder=3, rasterized=len(shaps) > 500)

                # plot the non-nan values colored by the trimmed feature value
                cvals = values[np.invert(nan_mask)].astype(np.float64)
                cvals_imp = cvals.copy()
                cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
                cvals[cvals_imp > vmax] = vmax
                cvals[cvals_imp < vmin] = vmin
                pl.scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
                           cmap=colors.red_blue, vmin=vmin, vmax=vmax, s=16,
                           c=cvals, alpha=alpha, linewidth=0,
                           zorder=3, rasterized=len(shaps) > 500)
            else:

                pl.scatter(shaps, pos + ys, s=16, alpha=alpha, linewidth=0, zorder=3,
                           color=color if colored_feature else "#777777", rasterized=len(shaps) > 500)
                
    # draw the color bar
    if color_bar and features is not None and plot_type != "bar" and \
            (plot_type != "layered_violin" or color in pl.cm.datad):
        import matplotlib.cm as cm
        m = cm.ScalarMappable(cmap=colors.red_blue if plot_type != "layered_violin" else pl.get_cmap(color))
        m.set_array([0, 1])
        cb = pl.colorbar(m, ticks=[0, 1], aspect=1000)
        cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
        for t in cb.ax.get_yticklabels():
            t.set_fontproperties(font)
            t.set_alpha(0.8)
        cb.set_label(color_bar_label, size=16, labelpad=0, alpha=0.8, fontproperties=font)
        cb.ax.tick_params(labelsize=14, length=0, grid_alpha=0.8)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
        cb.ax.set_aspect((bbox.height - 0.9) * 20)
        # cb.draw_all()

    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().spines['left'].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    pl.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=16, fontproperties=font)
    if plot_type != "bar":
        pl.gca().tick_params('y', length=20, width=0.5, which='major')
    #pl.gca().tick_params('x', labelsize=14)
    pl.xticks(ticks=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], labels=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], fontsize=14, fontproperties=font)
    pl.ylim(-1, len(feature_order))
    if plot_type == "bar":
        pl.xlabel(labels['GLOBAL_VALUE'], fontsize=16)
    #else:
        #pl.xlabel(labels['VALUE'], fontsize=16, loc='left')
        
    pl.text(x_position[0], y_position[0], 'SHAP value', fontsize=16, alpha=0.8, fontproperties=font)
    pl.text(x_position[1], y_position[1],  u'\u27F5  \u27F6', alpha=0.8, fontsize=40)
    pl.text(x_position[2], y_position[2], 'No-transition to type 2 diabetes      Transition to type 2 diabetes', fontsize=16, alpha=0.8, fontproperties=font)

    if store:
        plt.savefig(f'shap_summary_{pred_year}.png', dpi=600, bbox_inches='tight')
        plt.savefig(f'shap_summary_{pred_year}.pdf', dpi=600, bbox_inches='tight')
        plt.savefig(f'shap_summary_{pred_year}.tiff', dpi=600, bbox_inches='tight')

    if show:
        pl.show()

def bar(model, seed, pred_years=np.arange(1, 6), figsize=(16,6), max_display=11, store=False, order=Explanation.abs,
        clustering=None, clustering_cutoff=0.5,
        merge_cohorts=False, show_data="auto", show=True):
    """ Create a bar plot of a set of SHAP values.
    If a single sample is passed then we plot the SHAP values as a bar chart. If an
    Explanation with many samples is passed then we plot the mean absolute value for
    each feature column as a bar chart.
    Parameters
    ----------
    shap_values : shap.Explanation or shap.Cohorts or dictionary of shap.Explanation objects
        A single row of a SHAP Explanation object (i.e. shap_values[0]) or a multi-row Explanation
        object that we want to summarize.
    max_display : int
        The maximum number of bars to display.
    show : bool
        If show is set to False then we don't call the matplotlib.pyplot.show() function. This allows
        further customization of the plot by the caller after the bar() function is finished. 
    """
    fig, ax = plt.subplots(1, len(pred_years), figsize=figsize, facecolor='white')
    for pred_year in pred_years:
        features = np.load(f'shap_data/features_{model}_{seed}_{pred_year}.npy', allow_pickle=True)
        values = np.load(f'shap_data/values_{model}_{seed}_{pred_year}.npy', allow_pickle=True)
        op_history = np.load(f'shap_data/op_history_{model}_{seed}_{pred_year}.npy', allow_pickle=True)
        feature_names = np.load(f'shap_data/feature_names_{model}_{seed}_{pred_year}.npy', allow_pickle=True)
        partition_tree =  None
        values = np.array([values])

        # we show the data on auto only when there are no transforms
        if show_data == "auto":
            show_data = len(op_history) == 0

        # build our auto xlabel based on the transform history of the Explanation object
        xlabel = "SHAP value"
        for op in op_history:
            if op["name"] == "abs":
                xlabel = "|"+xlabel+"|"
            elif op["name"] == "__getitem__":
                pass # no need for slicing to effect our label, it will be used later to find the sizes of cohorts
            else:
                xlabel = str(op["name"])+"("+xlabel+")"

        num_features = min(max_display, len(values[0]))
        max_display = min(max_display, num_features)

        # iteratively merge nodes until we can cut off the smallest feature values to stay within
        # num_features without breaking a cluster tree
        orig_inds = [[i] for i in range(len(values[0]))]
        orig_values = values.copy()
        while True:
            feature_order = np.argsort(np.mean([np.argsort(convert_ordering(order, Explanation(values[i]))) for i in range(values.shape[0])], 0))
            if partition_tree is not None:

                # compute the leaf order if we were to show (and so have the ordering respect) the whole partition tree
                clust_order = sort_inds(partition_tree, np.abs(values).mean(0))

                # now relax the requirement to match the parition tree ordering for connections above clustering_cutoff
                dist = scipy.spatial.distance.squareform(scipy.cluster.hierarchy.cophenet(partition_tree))
                feature_order = get_sort_order(dist, clust_order, clustering_cutoff, feature_order)

                # if the last feature we can display is connected in a tree the next feature then we can't just cut
                # off the feature ordering, so we need to merge some tree nodes and then try again.
                if max_display < len(feature_order) and dist[feature_order[max_display-1],feature_order[max_display-2]] <= clustering_cutoff:
                    #values, partition_tree, orig_inds = merge_nodes(values, partition_tree, orig_inds)
                    partition_tree, ind1, ind2 = merge_nodes(np.abs(values).mean(0), partition_tree)
                    for i in range(len(values)):
                        values[:,ind1] += values[:,ind2]
                        values = np.delete(values, ind2, 1)
                        orig_inds[ind1] += orig_inds[ind2]
                        del orig_inds[ind2]
                else:
                    break
            else:
                break

        # here we build our feature names, accounting for the fact that some features might be merged together
        feature_inds = feature_order[:max_display]
        y_pos = np.arange(len(feature_inds), 0, -1)
        feature_names_new = []
        for pos,inds in enumerate(orig_inds):
            if len(inds) == 1:
                feature_names_new.append(feature_names[inds[0]])
            else:
                full_print = " + ".join([feature_names[i] for i in inds])
                if len(full_print) <= 40:
                    feature_names_new.append(full_print)
                else:
                    max_ind = np.argmax(np.abs(orig_values).mean(0)[inds])
                    feature_names_new.append(feature_names[inds[max_ind]] + " + %d other features" % (len(inds)-1))
        feature_names = feature_names_new

        # see how many individual (vs. grouped at the end) features we are plotting
        if num_features < len(values[0]):
            num_cut = np.sum([len(orig_inds[feature_order[i]]) for i in range(num_features-1, len(values[0]))])
            values[:,feature_order[num_features-1]] = np.sum([values[:,feature_order[i]] for i in range(num_features-1, len(values[0]))], 0)

        # build our y-tick labels
        yticklabels = []
        for i in feature_inds:
            if features is not None and show_data:
                yticklabels.append(format_value(features[i], "%0.03f") + " = " + feature_names[i])
            else:
                yticklabels.append(feature_names[i])
        if num_features < len(values[0]):
            yticklabels[-1] = "Sum of %d other features" % num_cut

        # if negative values are present then we draw a vertical line to mark 0, otherwise the axis does this for us...
        negative_values_present = np.sum(values[:,feature_order[:num_features]] < 0) > 0
        if negative_values_present:
            pl.axvline(0, 0, 1, color="#000000", linestyle="-", linewidth=1, zorder=1)

        # draw the bars
        patterns = (None, '\\\\', '++', 'xx', '////', '*', 'o', 'O', '.', '-')
        total_width = 0.7
        bar_width = total_width / len(values)
        feature_inds = feature_inds[:-1]
        y_pos = y_pos[:-1]
        for i in range(len(values)):
            ypos_offset = - ((i - len(values) / 2) * bar_width + bar_width / 2)
            ax[pred_year-1].barh(
                y_pos + ypos_offset, values[i,feature_inds],
                bar_width, align='center',
                color=['grey' for j in range(len(y_pos))],
            )

        ax[pred_year-1].get_yaxis().set_visible(False)
        ax[pred_year-1].set_xticks([0, 0.1, 0.2, 0.3, 0.4])
        ax[pred_year-1].set_xticklabels([0, 0.1, 0.2, 0.3, 0.4], fontsize=14, fontproperties=font)
        ax[pred_year-1].set_xlabel(xlabel, fontsize=14, fontproperties=font)
        ax[pred_year-1].set_xlim(0, 0.47)
        for i in range(max_display-1):
            if i > 1:
                color = 'black'
                x_pos = values[0, feature_inds[i]] + 0.01
            else:
                color = 'white'
                x_pos = values[0, feature_inds[2]] + 0.01
            ax[pred_year-1].text(x_pos, y_pos[i] + ypos_offset - 0.12, 
                                 rename_features[yticklabels[i]], color=color, 
                                 fontsize=14, fontproperties=font)
        #ax[pred_year-1].title.set_text(f'FH: {pred_year} years', fontproperties=font)
        if pred_year == 1:
            ax[pred_year-1].text(0.15, 12, f'FH: {pred_year} year', fontsize=16, fontproperties=font)
        else:
            ax[pred_year-1].text(0.13, 12, f'FH: {pred_year} years', fontsize=16, fontproperties=font)


    if show:
        pl.show()
    if store:
        plt.savefig(f'plots/shap_bar_{model}.png', dpi=600, bbox_inches='tight')

def rename_data(X):
    """
    Renames the predictors for the SHAP summary plot including all forecast horizons.
    """
    X_renamed = X.rename({'test=creatinineserum': 'SCr',
                          'test=hba1c(%)': 'HbA1c',
                          'test=glucose': 'Glucose', 
                          'test=calciumserum': 'SCa',
                          'test=sodiumserum': 'SSo',
                          'test=alt(gpt)': 'ALT',
                          'icd9_prefix=401_hist': 'ICD-9: 401',
                          'age': 'Age',
                          'sex': 'Sex', 
                          'bmi': 'BMI', 
                          'test=hdl-cholest.': 'HDL',
                          'test=triglycerides': 'TG',
                          'icd9_prefix=786': 'ICD-9: 786',
                          'icd9_prefix=719': 'ICD-9: 719',
                          'icd9_prefix=729': 'ICD-9: 729',
                          'icd9_prefix=278': 'ICD-9: 278',
                          'icd9_prefix=780': 'ICD-9: 780',
                          'icd9_prefix=782': 'ICD-9: 782',
                          '(med_class = 0)': 'Statins',
                          'icd9_prefix=724': 'ICD-9: 724',
                          'test=phosphorus': 'P',
                          'test=lymphocytes': 'LC',
                          'test=monocytes%': 'MC',
                          'test=uricacid': 'UA',
                          'test=tsh': 'TSH',
                          'test=bilirubintotal': 'TB',
                          'test=mpv': 'MPV', 
                          'test=alk.phosphat.': 'ALP',
                          'test=bun': 'BUN',
                          'test=mchc': 'MCHC',
                          'test=vitaminb12': 'Vitamin B12',
                          'test=basophils': 'BP',
                          'test=wbc': 'WBC'}, axis=1)
    return X_renamed

rename_features = {'test=creatinineserum': 'SCr',
                          'test=hba1c(%)': 'HbA1c',
                          'test=glucose': 'Glucose', 
                          'test=calciumserum': 'SCa',
                          'test=sodiumserum': 'SSo',
                          'test=alt(gpt)': 'ALT',
                          'icd9_prefix=401_hist': 'ICD-9: 401',
                          'age': 'Age',
                          'sex': 'Sex', 
                          'bmi': 'BMI', 
                          'test=hdl-cholest.': 'HDL',
                          'test=triglycerides': 'TG',
                          'icd9_prefix=786': 'ICD-9: 786',
                          'icd9_prefix=719': 'ICD-9: 719',
                          'icd9_prefix=729': 'ICD-9: 729',
                          'icd9_prefix=278': 'ICD-9: 278',
                          'icd9_prefix=780': 'ICD-9: 780',
                          'icd9_prefix=782': 'ICD-9: 782',
                          '(med_class = 0)': 'Statins',
                          'icd9_prefix=724': 'ICD-9: 724',
                          'test=phosphorus': 'P',
                          'test=lymphocytes': 'LC',
                          'test=monocytes%': 'MC',
                          'test=uricacid': 'UA',
                          'test=tsh': 'TSH',
                          'test=bilirubintotal': 'TB',
                          'test=mpv': 'MPV', 
                          'test=alk.phosphat.': 'ALP',
                          'test=bun': 'BUN',
                          'test=mchc': 'MCHC',
                          'test=vitaminb12': 'Vitamin B12',
                          'test=basophils': 'BP',
                          'test=wbc': 'WBC'}

def summary_plot_all(seed, model, pred_years=np.arange(1, 6), store=False, 
                     feature_names=None, max_display=10, plot_type=None,
                     color=None, axis_color="#333333", title=None, alpha=1, show=False, sort=True,
                     color_bar=True, plot_size="auto", layered_violin_max_num_bins=20, class_names=None,
                     class_inds=None,
                     color_bar_label=labels["FEATURE_VALUE"],
                     # depreciated
                     auto_size_plot=None,
                     use_log_scale=False):
    
    """
    Generates the SHAP summary plot including all five forecast horizons as in Figure 1B.
    """
    
    plt.rcParams['ytick.major.pad']=-16

    plt.figure(figsize=(18, 8), facecolor='white')
    for pred_year in pred_years:
        plt.subplot(1, 5, pred_year)
        shap_values = np.load(f'shap_data/shap_values_{model}_{seed}_{pred_year}.npy', allow_pickle=True)
        features_all = pd.read_csv(f'data/data_{pred_year}.csv', index_col=['id', 'year']).drop('y', axis=1)
        if model == 'wo_antidiabetic_medications':
            features_all = pd.read_csv(f'data/data_wo_antidiabetic_medications_{pred_year}.csv', index_col=['id', 'year'])
        if model != 'manual':
            cols = ['family_history', 'alcohol_abuse', 'smoking', 'pancreas_disease',
                    '(med_class = 20)', '(med_class = 21)', '(med_class = 22)', '(med_class = 23)']
            features_all = features_all.drop(cols, axis=1)
        if model == 'male' or model == 'female':
            if model == 'male':
                sex = 0
            elif model == 'female':
                sex=1
            features_all = features_all[features_all['sex'] == sex]
            features_all = features_all.drop('sex', axis=1)

        features = rename_data(features_all)
        multi_class = False
        if isinstance(shap_values, list):
            multi_class = True
            if plot_type is None:
                plot_type = "bar" # default for multi-output explanations
            assert plot_type == "bar", "Only plot_type = 'bar' is supported for multi-output explanations!"
        else:
            if plot_type is None:
                plot_type = "dot" # default for single output explanations
            assert len(shap_values.shape) != 1, "Summary plots need a matrix of shap_values, not a vector."

        # default color:
        if color is None:
            color = colors.blue_rgb

        # convert from a DataFrame or other types
        if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
            if feature_names is None:
                feature_names = features.columns
            features = features.values
        elif isinstance(features, list):
            if feature_names is None:
                feature_names = features
            features = None
        elif (features is not None) and len(features.shape) == 1 and feature_names is None:
            feature_names = features
            features = None

        num_features = (shap_values[0].shape[1] if multi_class else shap_values.shape[1])

        if feature_names is None:
            feature_names = np.array([labels['FEATURE'] % str(i) for i in range(num_features)])

        if use_log_scale:
            plt.xscale('symlog')


        if max_display is None:
            max_display = 20

        if sort:
            # order features by the sum of their effect magnitudes
            if multi_class:
                feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0))
            else:
                feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
            feature_order = feature_order[-min(max_display, len(feature_order)):]
        else:
            feature_order = np.flip(np.arange(min(max_display, num_features)), 0)

        row_height = 0.4
        plt.axvline(x=0, color="#999999", zorder=-1)

        if plot_type == "dot":
            for pos, i in enumerate(feature_order):
                plt.axhline(y=pos, color="grey", lw=0.8, dashes=(1, 5), zorder=-1)
                shaps = shap_values[:, i]
                values = None if features is None else features[:, i]
                inds = np.arange(len(shaps))
                np.random.shuffle(inds)
                if values is not None:
                    values = values[inds]
                shaps = shaps[inds]
                colored_feature = True
                try:
                    values = np.array(values, dtype=np.float64)  # make sure this can be numeric
                except:
                    colored_feature = False
                N = len(shaps)
                # hspacing = (np.max(shaps) - np.min(shaps)) / 200
                # curr_bin = []
                nbins = 100
                quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
                inds = np.argsort(quant + np.random.randn(N) * 1e-6)
                layer = 0
                last_bin = -1
                ys = np.zeros(N)
                for ind in inds:
                    if quant[ind] != last_bin:
                        layer = 0
                    ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                    layer += 1
                    last_bin = quant[ind]
                ys *= 0.9 * (row_height / np.max(ys + 1))

                if features is not None and colored_feature:
                    # trim the color range, but prevent the color range from collapsing
                    vmin = np.nanpercentile(values, 5)
                    vmax = np.nanpercentile(values, 95)
                    if vmin == vmax:
                        vmin = np.nanpercentile(values, 1)
                        vmax = np.nanpercentile(values, 99)
                        if vmin == vmax:
                            vmin = np.min(values)
                            vmax = np.max(values)
                    if vmin > vmax: # fixes rare numerical precision issues
                        vmin = vmax

                    assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"

                    # plot the nan values in the interaction feature as grey
                    nan_mask = np.isnan(values)
                    plt.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777", vmin=vmin,
                               vmax=vmax, s=16, alpha=alpha, linewidth=0,
                               zorder=3, rasterized=len(shaps) > 500)

                    # plot the non-nan values colored by the trimmed feature value
                    cvals = values[np.invert(nan_mask)].astype(np.float64)
                    cvals_imp = cvals.copy()
                    cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
                    cvals[cvals_imp > vmax] = vmax
                    cvals[cvals_imp < vmin] = vmin
                    plt.scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
                               cmap=colors.red_blue, vmin=vmin, vmax=vmax, s=16,
                               c=cvals, alpha=alpha, linewidth=0,
                               zorder=3, rasterized=len(shaps) > 500)
                else:

                    plt.scatter(shaps, pos + ys, s=16, alpha=alpha, linewidth=0, zorder=3,
                               color=color if colored_feature else "#777777", rasterized=len(shaps) > 500)


        plt.gca().xaxis.set_ticks_position('bottom')
        plt.gca().yaxis.set_ticks_position('none')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().tick_params(color='black', labelcolor='black')
        plt.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=14,
                  ha='right', fontproperties=font)
        if plot_type != "bar":
            plt.gca().tick_params('y', length=20, width=0.5, which='major')
        #plt.gca().tick_params('x', labelsize=14)
        #plt.xticks(ticks=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], labels=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], fontsize=14)
        plt.ylim(-1, len(feature_order))
        plt.xticks([0, 1], fontsize=14, fontproperties=font)

        plt.xlabel('SHAP value', fontsize=14, alpha=1, fontproperties=font)
        if pred_year == 1:
            plt.title(f'FH: {pred_year} year', fontsize=16, loc='left', fontproperties=font)
        else:
            plt.title(f'FH: {pred_year} years', fontsize=16, loc='left', fontproperties=font)
    
    # Draw color bar
    import matplotlib.cm as cm
    m = cm.ScalarMappable(cmap=colors.red_blue if plot_type != "layered_violin" else plt.get_cmap(color))
    m.set_array([0, 1])
    cb = plt.colorbar(m, ticks=[0, 1], aspect=10)
    cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
    for t in cb.ax.get_yticklabels():
        t.set_fontproperties(font)
        t.set_alpha(0.8)
    cb.set_label(color_bar_label, size=14, labelpad=1, alpha=1, fontproperties=font)
    cb.ax.tick_params(labelsize=14, length=0, grid_alpha=1, pad=5)
    cb.set_alpha(1)
    cb.outline.set_visible(False)
    bbox = cb.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
    cb.ax.set_aspect((bbox.height - 0.9) * 20)
    
    
    plt.subplots_adjust(wspace=0.54, hspace=0.25)
    if show:
        plt.show()
        
    if store:
        plt.savefig(f'plots/shap_summary.png', dpi=600, bbox_inches='tight')
