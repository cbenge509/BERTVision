############################################################################
# IMPORTS
############################################################################

import numpy as np
import pandas as pd
import altair as alt
from collections import OrderedDict


############################################################################
# Plotting Utilities, Constants, Methods for W209 arXiv project
############################################################################

#---------------------------------------------------------------------------
## Plotting Palette
#
# Create a dict object containing U.C. Berkeley official school colors for plot palette 
# reference : https://brand.berkeley.edu/colors/
# secondary reference : https://alumni.berkeley.edu/brand/color-palette# CLass Initialization
#---------------------------------------------------------------------------

berkeley_palette = OrderedDict({
                    'berkeley_blue'     : '#003262',
                    'california_gold'   : '#fdb515',
                    'founders_rock'     : '#3b7ea1',
                    'medalist'          : '#c4820e',
                    'bay_fog'           : '#ddd5c7',
                    'lawrence'          : '#00b0da',
                    'sather_gate'       : '#b9d3b6',
                    'pacific'           : '#46535e',
                    'soybean'           : '#859438',
                    'south_hall'        : '#6c3302',
                    'wellman_tile'      : '#D9661F',
                    'rose_garden'       : '#ee1f60',
                    'golden_gate'       : '#ed4e33',
                    'lap_lane'          : '#00a598',
                    'ion'               : '#cfdd45',
                    'stone_pine'        : '#584f29',
                    'grey'              : '#eeeeee',
                    'web_grey'          : '#888888',
                    # alum only colors
                    'metallic_gold'     : '#BC9B6A',
                    'california_purple' : '#5C3160'                   
                    }
                    )

#---------------------------------------------------------------------------
## Altair custom "Cal" theme
#---------------------------------------------------------------------------

def cal_theme():
    font = "Lato"

    return {
        "config": {
            "title": {
                "fontSize": 30,
                "font": font,
                "anchor": "middle",
                "align":"center",
                "color": berkeley_palette['berkeley_blue'],
                "subtitleFontSize": 20,
                "subtitleFont": font,
                "subtitleAcchor": "middle",
                "subtitleAlign": "center",
                "subtitleColor": berkeley_palette['berkeley_blue']
            },
            "axisX": {
                "labelFont": font,
                "labelColor": berkeley_palette['pacific'],
                "labelFontSize": 15,
                "titleFont": font,
                "titleFontSize": 20,
                "titleColor": berkeley_palette['berkeley_blue'],
                "titleAlign": "right",
                "titleAnchor": "end",
                "titlePadding": 20
            },
            "axisY": {
                "labelFont": font,
                "labelColor": berkeley_palette['pacific'],
                "labelFontSize": 15,
                "titleFont": font,
                "titleFontSize": 20,
                "titleColor": berkeley_palette['berkeley_blue'],
                "titleAlign": "right",
                "titleAnchor": "end",
                "titlePadding": 20
            },
            "headerRow": {
                "labelFont": font,
                "titleFont": font,
                "titleFontSize": 15,
                "titleColor": berkeley_palette['berkeley_blue'],
                "titleAlign": "right",
                "titleAnchor": "end"
            },
            "legend": {
                "labelFont": font,
                "labelFontSize": 15,
                "labelColor": berkeley_palette['stone_pine'],
                "symbolType": "stroke",
                "symbolStrokeWidth": 3,
                "symbolOpacity": 1.0,
                "symbolSize": 500,
                "titleFont": font,
                "titleFontSize": 20,
                "titleColor": berkeley_palette['berkeley_blue']
            },
            "view": {
                "labelFont": font,
                "labelColor": berkeley_palette['pacific'],
                "labelFontSize": 15,
                "titleFont": font,
                "titleFontSize": 20,
                "titleColor": berkeley_palette['berkeley_blue'],
                "titleAlign": "right",
                "titleAnchor": "end"
            },
            "facet": {
                "labelFont": font,
                "labelColor": berkeley_palette['pacific'],
                "labelFontSize": 15,
                "titleFont": font,
                "titleFontSize": 20,
                "titleColor": berkeley_palette['berkeley_blue'],
                "titleAlign": "right",
                "titleAnchor": "end"
            },
            "row": {
                "labelFont": font,
                "labelColor": berkeley_palette['pacific'],
                "labelFontSize": 15,
                "titleFont": font,
                "titleFontSize": 20,
                "titleColor": berkeley_palette['berkeley_blue'],
                "titleAlign": "right",
                "titleAnchor": "end"
            }

        }
    }

alt.themes.register("my_cal_theme", cal_theme)
alt.themes.enable("my_cal_theme")


#---------------------------------------------------------------------------
## Altair : Frozen Weights Performance plot
##
## assumes data in shape of :
##      Frozen Weights Pct     1        2        3        4
##      ------------------     -------- -------- -------- --------
##      0.33845...             0.773913 0.811014 0.812754 0.818551
##      ...
#---------------------------------------------------------------------------

def altair_frozen_weights_performance_plot(data, xaxis_title = "Frozen Weights Pct", yaxis_title = "Dev Metric", width = 1200, height = 600,
    title_main = "Dense Variably Unfrozen", title_subtitle = "GLUE Task: MSR", comparison_bert_type = "BERT-base", 
    ci_bar = True, comparison_bert_range = [0.842, 0.830, 0.828, 0.827, 0.843], line_type = "poly", poly_order = 10, 
    line_color_range = [berkeley_palette['berkeley_blue'], berkeley_palette['wellman_tile'], berkeley_palette['rose_garden'], berkeley_palette['lawrence']],
    ci_bar_color = berkeley_palette['golden_gate'], AdapterBERT_performance = None):

    assert type(data) is pd.core.frame.DataFrame, "Parameter `data` must be of type pandas.core.frame.DataFrame."
    assert all(e in data.columns.to_list() for e in ['Frozen Weights Pct', '1', '2', '3', '4']), "Parameter `data` must contain the following columns: ['Frozen Weights Pct', '1', '2', '3', '4']."

    if ci_bar:
        assert type(comparison_bert_range) is list, "Parameter `comparison_bert_range` must be of type list."
        assert len(comparison_bert_range) > 1, "Parameter `comparison_bert_range` must contain at least two values."
        assert all(isinstance(x, float) for x in comparison_bert_range), "All values in `comparison_bert_range` must be of type float."
    
    assert line_type in ['poly', 'loess'], "Parameter `line_type` only supports the options: 'poly' or 'loess'."
    if line_type == 'poly':
        assert type(poly_order) is int, "Parameter `poly_order` must be of type int."
        assert 0 < poly_order <= 20, "Parameter `poly_order` must be integer value between 1 and 20."

    if AdapterBERT_performance is not None:
        assert type(AdapterBERT_performance) is float, "Parameter `AdapterBERT_performance` must be of type None or type float."

    band_range_ = list(np.arange(0.0,1.1,0.1))
    y_lower_, y_upper_ = min(np.hstack([data['1'], data['2'], data['3'], data['4']])), max(np.hstack([data['1'], data['2'], data['3'], data['4'], np.where(np.isnan(AdapterBERT_performance),0,AdapterBERT_performance)]))
    x_lower_, x_upper_ = min(data['Frozen Weights Pct'] - 0.05), max(data['Frozen Weights Pct'] + 0.05)
    bert_base_avg_ = " ".join([comparison_bert_type, "baseline:", str(round(np.mean(comparison_bert_range), 5))])
    ab_perf_avg_ = " ".join(["AdapterBERT baseline:", str(round(np.mean(np.where(np.isnan(AdapterBERT_performance),0,AdapterBERT_performance)), 5)), "(64 Adapters)"])

    base = alt.Chart(data).mark_point(opacity=0.3, size=30)\
        .transform_fold(
            fold=['1','2','3','4'],
            as_=['category', 'Dev Metric']
        ).encode(
            x=alt.X('Frozen Weights Pct:Q', scale=alt.Scale(domain=[x_lower_, x_upper_])), 
            y=alt.Y('Dev Metric:Q', scale=alt.Scale(domain=[y_lower_, y_upper_])), 
            color=alt.Color('category:N', scale= alt.Scale(range = line_color_range),
                legend=alt.Legend(title='Epochs', symbolOpacity=1.0))
        ).properties(width = 1200, height = 600)

    if line_type == "poly":
        reg = base.transform_regression('Frozen Weights Pct', 'Dev Metric', method='poly', groupby=['category'], order = poly_order)\
            .mark_line(size=5, opacity=1.0)
    else:
        reg = base.transform_loess('Frozen Weights Pct','Dev Metric', groupby=['category'])\
            .mark_line(size=5, opacity=1.0)

    line = alt.Chart(pd.DataFrame({'Dev Metric': [np.mean(comparison_bert_range)]}))\
        .mark_rule(size=3, strokeDash=[8,5], color=berkeley_palette['pacific']).encode(y='Dev Metric')

    text = alt.Chart(pd.DataFrame({'Frozen Weights Pct':[0.8], 'Dev Metric':[np.max(comparison_bert_range) + 0.006], 'out':[bert_base_avg_]}))\
        .mark_text(fontSize=18, font='Lato', color=berkeley_palette['pacific'])\
            .encode(text='out',x='Frozen Weights Pct:Q',y='Dev Metric:Q')

    ab_perf = None
    if AdapterBERT_performance is not None:
        ab_perf = alt.Chart(pd.DataFrame({'Dev Metric': [AdapterBERT_performance]}))\
           .mark_rule(size=3, strokeDash=[8,5], color=berkeley_palette['lap_lane'])\
               .encode(y='Dev Metric')
        
        ab_text = alt.Chart(pd.DataFrame({'Frozen Weights Pct':[0.2], 'Dev Metric':[np.max([np.max(comparison_bert_range) + 0.006, AdapterBERT_performance + 0.010])], 'out':[ab_perf_avg_]}))\
            .mark_text(fontSize=18, font='Lato', color=berkeley_palette['pacific'])\
                .encode(text='out',x='Frozen Weights Pct:Q',y='Dev Metric:Q')

    if ci_bar:
        band = alt.Chart(pd.DataFrame({'x':band_range_, 'lower':[min(comparison_bert_range)] * len(band_range_), 'upper':[max(comparison_bert_range)] * len(band_range_)}))\
            .mark_area(opacity = 0.2, color=ci_bar_color).encode(
                x=alt.X('x', axis=alt.Axis(title=xaxis_title)),
                y=alt.Y('lower', axis=alt.Axis(title=yaxis_title)),
                y2='upper')

        if ab_perf is not None:
            viz = alt.layer(base + reg + line + text + band + ab_perf + ab_text).configure_view(strokeWidth=0)\
                .configure_axis(grid=False)\
                .properties(title = {"text" : title_main, "subtitle" : title_subtitle})
        else:
            viz = alt.layer(base + reg + line + text + band).configure_view(strokeWidth=0)\
                .configure_axis(grid=False)\
                .properties(title = {"text" : title_main, "subtitle" : title_subtitle})

    else:
        if ab_perf is not None:
            viz = alt.layer(base + reg + line + text + ab_perf + ab_text).configure_view(strokeWidth=0)\
                .configure_axis(grid=False)\
                .properties(title = {"text" : title_main, "subtitle" : title_subtitle})
        else:
            viz = alt.layer(base + reg + line + text).configure_view(strokeWidth=0)\
                .configure_axis(grid=False)\
                .properties(title = {"text" : title_main, "subtitle" : title_subtitle})


    return viz

#---------------------------------------------------------------------------
## Altair : Frozen Weights Performance plot (for 8 lines)
##
## assumes data in shape of :
##      Frozen Weights Pct     1        2        3        4        5        6        7        8
##      ------------------     -------- -------- -------- -------- -------- -------- -------- --------
##      0.33845...             0.773913 0.811014 0.812754 0.818551 0.773913 0.811014 0.812754 0.818551
##      ...
#---------------------------------------------------------------------------

def altair_frozen_weights_performance_plot8(data, xaxis_title = "Frozen Weights Pct", yaxis_title = "Dev Metric", width = 1200, height = 600,
    title_main = "Dense Variably Unfrozen", title_subtitle = "GLUE Task: MSR", comparison_bert_type = "BERT-base", 
    ci_bar = True, comparison_bert_range = [0.842, 0.830, 0.828, 0.827, 0.843, 0.842, 0.830, 0.828, 0.827, 0.843], line_type = "poly", poly_order = 10, 
    line_color_range = [berkeley_palette['berkeley_blue'], berkeley_palette['wellman_tile'], berkeley_palette['rose_garden'], berkeley_palette['lawrence'],
        berkeley_palette['lap_lane'],berkeley_palette['medalist'],berkeley_palette['south_hall'],berkeley_palette['soybean']],
    ci_bar_color = berkeley_palette['golden_gate'], AdapterBERT_performance = None):

    assert type(data) is pd.core.frame.DataFrame, "Parameter `data` must be of type pandas.core.frame.DataFrame."
    assert all(e in data.columns.to_list() for e in ['Frozen Weights Pct', '1', '2', '3', '4', '5', '6', '7', '8']), "Parameter `data` must contain the following columns: ['Frozen Weights Pct', '1', '2', '3', '4', '5', '6', '7', '8']."

    if ci_bar:
        assert type(comparison_bert_range) is list, "Parameter `comparison_bert_range` must be of type list."
        assert len(comparison_bert_range) > 1, "Parameter `comparison_bert_range` must contain at least two values."
        assert all(isinstance(x, float) for x in comparison_bert_range), "All values in `comparison_bert_range` must be of type float."
    
    assert line_type in ['poly', 'loess'], "Parameter `line_type` only supports the options: 'poly' or 'loess'."
    if line_type == 'poly':
        assert type(poly_order) is int, "Parameter `poly_order` must be of type int."
        assert 0 < poly_order <= 20, "Parameter `poly_order` must be integer value between 1 and 20."

    if AdapterBERT_performance is not None:
        assert type(AdapterBERT_performance) is float, "Parameter `AdapterBERT_performance` must be of type None or type float."

    band_range_ = list(np.arange(0.0,1.1,0.1))
    y_lower_, y_upper_ = min(np.hstack([data['1'], data['2'], data['3'], data['4'], data['5'], data['6'], data['7'], data['8']])), max(np.hstack([data['1'], data['2'], data['3'], data['4'], data['5'], data['6'], data['7'], data['8'], np.where(np.isnan(AdapterBERT_performance),0,AdapterBERT_performance)]))
    x_lower_, x_upper_ = min(data['Frozen Weights Pct'] - 0.05), max(data['Frozen Weights Pct'] + 0.05)
    bert_base_avg_ = " ".join([comparison_bert_type, "baseline:", str(round(np.mean(comparison_bert_range), 5))])
    ab_perf_avg_ = " ".join(["AdapterBERT baseline:", str(round(np.mean(np.where(np.isnan(AdapterBERT_performance),0,AdapterBERT_performance)), 5)), "(64 Adapters)"])

    base = alt.Chart(data).mark_point(opacity=0.3, size=30)\
        .transform_fold(
            fold=['1','2','3','4', '5', '6', '7', '8'],
            as_=['category', 'Dev Metric']
        ).encode(
            x=alt.X('Frozen Weights Pct:Q', scale=alt.Scale(domain=[x_lower_, x_upper_])), 
            y=alt.Y('Dev Metric:Q', scale=alt.Scale(domain=[y_lower_, y_upper_])), 
            color=alt.Color('category:N', scale= alt.Scale(range = line_color_range),
                legend=alt.Legend(title='Epochs', symbolOpacity=1.0))
        ).properties(width = 1200, height = 600)

    if line_type == "poly":
        reg = base.transform_regression('Frozen Weights Pct', 'Dev Metric', method='poly', groupby=['category'], order = poly_order)\
            .mark_line(size=5, opacity=1.0)
    else:
        reg = base.transform_loess('Frozen Weights Pct','Dev Metric', groupby=['category'])\
            .mark_line(size=5, opacity=1.0)

    line = alt.Chart(pd.DataFrame({'Dev Metric': [np.mean(comparison_bert_range)]}))\
        .mark_rule(size=3, strokeDash=[8,5], color=berkeley_palette['pacific']).encode(y='Dev Metric')

    text = alt.Chart(pd.DataFrame({'Frozen Weights Pct':[0.8], 'Dev Metric':[np.max(comparison_bert_range) + 0.006], 'out':[bert_base_avg_]}))\
        .mark_text(fontSize=18, font='Lato', color=berkeley_palette['pacific'])\
            .encode(text='out',x='Frozen Weights Pct:Q',y='Dev Metric:Q')

    ab_perf = None
    if AdapterBERT_performance is not None:
        ab_perf = alt.Chart(pd.DataFrame({'Dev Metric': [AdapterBERT_performance]}))\
           .mark_rule(size=3, strokeDash=[8,5], color=berkeley_palette['lap_lane'])\
               .encode(y='Dev Metric')
        
        ab_text = alt.Chart(pd.DataFrame({'Frozen Weights Pct':[0.2], 'Dev Metric':[np.max([np.max(comparison_bert_range) + 0.006, AdapterBERT_performance + 0.010])], 'out':[ab_perf_avg_]}))\
            .mark_text(fontSize=18, font='Lato', color=berkeley_palette['pacific'])\
                .encode(text='out',x='Frozen Weights Pct:Q',y='Dev Metric:Q')

    if ci_bar:
        band = alt.Chart(pd.DataFrame({'x':band_range_, 'lower':[min(comparison_bert_range)] * len(band_range_), 'upper':[max(comparison_bert_range)] * len(band_range_)}))\
            .mark_area(opacity = 0.2, color=ci_bar_color).encode(
                x=alt.X('x', axis=alt.Axis(title=xaxis_title)),
                y=alt.Y('lower', axis=alt.Axis(title=yaxis_title)),
                y2='upper')

        if ab_perf is not None:
            viz = alt.layer(base + reg + line + text + band + ab_perf + ab_text).configure_view(strokeWidth=0)\
                .configure_axis(grid=False)\
                .properties(title = {"text" : title_main, "subtitle" : title_subtitle})
        else:
            viz = alt.layer(base + reg + line + text + band).configure_view(strokeWidth=0)\
                .configure_axis(grid=False)\
                .properties(title = {"text" : title_main, "subtitle" : title_subtitle})

    else:
        if ab_perf is not None:
            viz = alt.layer(base + reg + line + text + ab_perf + ab_text).configure_view(strokeWidth=0)\
                .configure_axis(grid=False)\
                .properties(title = {"text" : title_main, "subtitle" : title_subtitle})
        else:
            viz = alt.layer(base + reg + line + text).configure_view(strokeWidth=0)\
                .configure_axis(grid=False)\
                .properties(title = {"text" : title_main, "subtitle" : title_subtitle})


    return viz

#---------------------------------------------------------------------------
## Altair : Frozen Weights per Epoch Ridge Plot
##
## assumes data in shape of :
##      Frozen Weights Pct     Epoch    Dev Metric
##      ------------------     -------- -------------
##      0.5                    1        0.811014
##      0.3                    2        0.650343
##      ...
#---------------------------------------------------------------------------

def altair_frozen_weights_performance_ridge_plot(data, xaxis_title = "Dev Metric", title_main = "Dense Variably Unfrozen", task_name = "MSR", 
    step_all = 75, width_all = 600, step_small = 30, width_small = 400, overlap = 1, max_bins = 30, color_scheme = 'redyellowblue', return_all = True):

    assert type(data) is pd.core.frame.DataFrame, "Parameter `data` must be of type pandas.core.frame.DataFrame."
    assert all(e in data.columns.to_list() for e in ['Frozen Weights Pct', 'Epoch', 'Dev Metric']), "Parameter `data` must contain the following columns: ['Frozen Weights Pct', 'Epoch', 'Dev Metric']."

    # generate the combined epochs plot
    domain_ = [min(data['Dev Metric']), max(data['Dev Metric'])]
    c0 = alt.Chart(data, height=step_all)\
        .transform_joinaggregate(mean_acc='mean(Dev Metric)', groupby=['Frozen Weights Pct'])\
        .transform_bin(['bin_max', 'bin_min'], 'Dev Metric', bin=alt.Bin(maxbins=max_bins))\
        .transform_aggregate(value='count()', groupby=['Frozen Weights Pct', 'mean_acc', 'bin_min', 'bin_max'])\
        .transform_impute(impute='value', groupby=['Frozen Weights Pct', 'mean_acc'], key='bin_min', value=domain_[0])\
        .mark_area(interpolate='monotone', fillOpacity=0.8, stroke='lightgray', strokeWidth=0.5)\
        .encode(
            alt.X('bin_min:Q', bin='binned', title=xaxis_title, scale=alt.Scale(domain=domain_)),
            alt.Y('value:Q', scale=alt.Scale(range=[step_all, -step_all * overlap]), axis=None),
            alt.Fill('mean_acc:Q', legend=None,scale=alt.Scale(domain=[sum(x) for x in zip(domain_[::-1], [-0.05, 0.05])], scheme=color_scheme)))\
        .properties(width = width_all, height = step_all)\
        .facet(
            row=alt.Row(
                'Frozen Weights Pct:O',
                title='Forzen Weights Pct (Binned)',
                header=alt.Header(
                    labelAngle=0, labelAlign='right', labelFontSize=15, labelFont='Lato', labelColor=berkeley_palette['pacific'], titleFontSize=20
                )
            )
        ).properties(title={'text':title_main, 'subtitle': " ".join([task_name,"- All Epochs"])}, bounds='flush')
        

    # if not returning all plots, then return the main "All Epochs" plot
    if not (return_all):
        return c0.configure_facet(spacing=0).configure_view(stroke=None).configure_title(anchor='middle')
    
    # generate the individual epochs plots
    subplots = [None] * 4
    for i in range(1,5):

        domain_ = [min(data[(data['Epoch'] == i)]['Dev Metric']), max(data[(data['Epoch'] == i)]['Dev Metric'])]

        o = alt.Chart(data[(data['Epoch'] == i)], height=step_small)\
            .transform_joinaggregate(mean_acc='mean(Dev Metric)', groupby=['Frozen Weights Pct'])\
            .transform_bin(['bin_max', 'bin_min'], 'Dev Metric', bin=alt.Bin(maxbins=max_bins))\
            .transform_aggregate(value='count()', groupby=['Frozen Weights Pct', 'mean_acc', 'bin_min', 'bin_max'])\
            .transform_impute(impute='value', groupby=['Frozen Weights Pct', 'mean_acc'], key='bin_min', value=domain_[0])\
            .mark_area(interpolate='monotone', fillOpacity=0.8, stroke='lightgray', strokeWidth=0.5)\
            .encode(
                alt.X('bin_min:Q', bin='binned', title=xaxis_title, scale=alt.Scale(domain=domain_)),
                alt.Y('value:Q', scale=alt.Scale(range=[step_small, -step_small * overlap]), axis=None),
                alt.Fill('mean_acc:Q', legend=None, scale=alt.Scale(domain=[sum(x) for x in zip(domain_[::-1], [-0.05, 0.05])], scheme=color_scheme)))\
            .properties(width = width_small, height = step_small)\
            .facet(
                row=alt.Row(
                    'Frozen Weights Pct:O',
                    title='Forzen Weights Pct (Binned)',
                    header=alt.Header(
                        labelAngle=0, labelAlign='right', labelFontSize=15, labelFont='Lato', labelColor=berkeley_palette['pacific'], titleFontSize=20
                    )
                )
            ).properties(title={'text':title_main, 'subtitle': " ".join([task_name, "- Epoch", str(i)])}, bounds='flush')

        subplots[i-1] = o

    viz = alt.hconcat(alt.vconcat(alt.hconcat(subplots[0], subplots[1]), alt.hconcat(subplots[2], subplots[3])), c0)\
        .configure_facet(spacing=0)\
        .configure_view(stroke=None)\
        .configure_title(anchor='middle')

    return viz

#---------------------------------------------------------------------------
## Altair : Comparison of Parasite to BERT faceted bar chart
##
## assumes data in shape of :
##      task     model        score
##      -------  -----------  -------------
##      QNLI     BERT-base    0.8194
##      SST-B    Parasite 16  0.650343
##      ...
##
## assumes number of model_color_range values equals number of unique models
#---------------------------------------------------------------------------
def altair_parasite_comparison_faceted_bar(data, yaxis_title = "Performance", title_main = "Parasite Model Performance", 
    subtitle = "Compared to BERT-base", height=600, width=200, model_color_range = [berkeley_palette['wellman_tile'], berkeley_palette['berkeley_blue'], berkeley_palette['pacific']]):

    assert type(data) is pd.core.frame.DataFrame, "Parameter `data` must be of type pandas.core.frame.DataFrame."
    assert all(e in data.columns.to_list() for e in ['task', 'model', 'score']), "Parameter `data` must contain the following columns: ['task','model','score']."
    assert len(np.unique(data.model)) == len(model_color_range), "Number of `model_color_range` values must match the number of unique models in the dataframe."

    # hah! I see the lazy color scheming going on here!  Trying to get out of explicitly providing them!  #Shame!
    #if len(np.unique(data.model)) > len(berkeley_palette):
    #    raise RuntimeError("Too many models to color code")
    #else:
    #    model_color_range = [berkeley_palette[color] for color in list(berkeley_palette.keys())[:len(berkeley_palette)]]

    base = alt.Chart().mark_bar(opacity=0.9).encode(
        #x=alt.X('model:N', axis=alt.Axis(title=None, labelFontSize=20, labelAngle=-45)),
        x=alt.X('model:N', axis=None),
        y=alt.Y('score:Q', axis=alt.Axis(title=yaxis_title, labelFontSize=20, titleFontSize=25), scale=alt.Scale(domain=[0.0, 1.0])),
        color=alt.Color('model:N', scale= alt.Scale(range = model_color_range), 
            legend=alt.Legend(title="Models", symbolType="square", labelFontSize=20, titlePadding=20, titleFontSize=25, labelLimit=500, orient='bottom'))
    ).properties(height=height, width=width)

    text = alt.Chart().mark_text(color='white', dy=15, size=15).encode(
        x=alt.X('model:N', axis=alt.Axis(title=None)),
        y=alt.Y('score:Q', axis=alt.Axis(title=yaxis_title), scale=alt.Scale(domain=[0.0, 1.0])),
        text=alt.Text('score:Q', format='.3N')
    )

    viz = alt.layer(base + text, data = data)\
        .facet(
            column=alt.Column('task:N', title=None, 
                header=alt.Header(labelAngle=0, labelAlign='center', labelFontSize=20, labelFont='Lato', labelColor=berkeley_palette['pacific'], titleFontSize=20, labelPadding=10)
            )
        ).configure(padding={'top':20, 'bottom':20, 'right':20, 'left':20})\
        .properties(title={'text':title_main, 'subtitle': subtitle}, bounds='flush')\
        .configure_title(dy=-10)
    
    return viz