import streamlit as st
import numpy as np
import pandas as pd
import gc

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import time
import datetime
import logging
import os, psutil
from typing import Union

st.set_page_config(page_title='Grad Stats',
         page_icon=':coffee:',
         layout="wide")

# Wanna just execute once
@st.cache
def log_to_file():
    fh = logging.FileHandler('gradstats.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger = st.logger.get_logger('gradstats_logger')
    logger.handlers = []
    logger.addHandler(fh)
    return logger

def vec_dt_replace(series, year=None, month=None, day=None):
    return pd.to_datetime(
        {'year': series.dt.year if year is None else year,
         'month': series.dt.month if month is None else month,
         'day': series.dt.day if day is None else day})

def filter_by(df, by: Union[str, dict], value: Union[str, list] = None, exact: bool = False):
    if by is None:
        return df

    filt = [True] * len(df)
    if isinstance(by, str):
        by = {by: (value, exact)}
    for k, v in by.items():
        value = v[0]
        exact = v[1]
        if value is None:
            continue
        if isinstance(value, str):
            if exact:
                filt = filt & (df[k] == value)
            else:
                filt = filt & (df[k].str.contains(value, case=False, regex=False))
        elif isinstance(value, list) and len(value) > 0:
            filt = filt & (df[k].isin(value))
    return filt

def create_filter(df,
                  degree: str = None,
                  decisionfin: Union[str, list] = None,
                  institution: Union[str, list] = None,
                  major: Union[str, list] = None,
                  status: Union[str, list] = None,
                  season: Union[str, list] = None):
    filter_dict = {}
    filter_dict['degree'] = (degree, True)
    filter_dict['institution'] = (institution, False)
    filter_dict['decisionfin'] = (decisionfin, True)
    filter_dict['major'] = (major, True)
    filter_dict['status'] = (status, True)
    filter_dict['season'] = (season, True)

    return filter_by(df, filter_dict)

def get_uni_stats(u_df,
                    search: str = None,
                    title: str = None,
                    degree: str = 'PhD',
                    field: str = None,
                    status: list = None,
                    seasons: list = None,
                    stack: bool = False,
                    axis_lines: bool = False,
                    grid_lines: bool = False,
                    hue: str = 'decisionfin',
                    debug: bool = False):
    if debug:
        start_time = time.time()
    if isinstance(degree, str) and degree not in ['MS', 'PhD', 'MEng', 'MFA', 'MBA', 'Other']:
        degree = 'PhD'

    gpa_multiple = 'stack' if stack else 'group'


    # Trying to pick red/green colorblind-friendly colors
    # 1. light green,
    # 2. red,
    # 3. dark blue
    color_dict = {
        'Accepted': '#2eff71',
        'Rejected': '#ff0000',
        'Interview': '#0000ff'
    }

    if debug:
        # We're logging at an info level because of streamlits logger...
        logger.info("prep time: %.2f seconds" % (time.time() - start_time))
        start_time = time.time()

    # This generates 4 graphs, so let's make it a 2x2 grid
    pltly_fig = make_subplots(rows=2, cols=2,
                              vertical_spacing=0.12,
                              subplot_titles=("Decision Timeline", "GPA Distribution", "GRE Score Distribution", "GRE AWA Score Distribution"))
    
    # Timeline stats
    u_df = u_df[create_filter(u_df, degree, ['Accepted', 'Rejected', 'Interview'], search, field, status, seasons)]
    if len(u_df) < 1:
        return pltly_fig

    if debug:
        logger.info("filter time: %.2f seconds" % (time.time() - start_time))
        start_time = time.time()

    graph_filt =  u_df['uniform_dates'] < datetime.datetime(2020, 6, 1)
    if sum(graph_filt) == 0:
        return pltly_fig

    if debug:
        logger.info("timeline aesthetics time: %.2f seconds" % (time.time() - start_time))
        start_time = time.time()

    for category in sorted(u_df[hue].unique().tolist()):
        grouped_df = u_df[(graph_filt) & (u_df[hue] == category)].groupby(by=['uniform_dates']).size().reset_index(name='counts')
        pltly_fig.add_trace(go.Scatter(x=grouped_df['uniform_dates'],
                                       y=grouped_df['counts'].cumsum(),
                                         name=category,
                                         mode='lines+markers',
                                         marker_color=color_dict[category]),
                            row=1,
                            col=1)
    
    # Get GPA stats
    gpa_filt = (~u_df['gpafin'].isna()) & (u_df['gpafin'] <= 4)
    gpa_bins = 10 if gpa_multiple == 'group' else 20
    if debug:
        logger.info("gpa time: %.2f seconds" % (time.time() - start_time))
        start_time = time.time()

    for category in sorted(u_df[hue].unique().tolist(), reverse=True):
        pltly_fig.add_trace(go.Histogram(x=u_df[(gpa_filt) & (u_df[hue] == category)]['gpafin'],
                                         nbinsx=gpa_bins,
                                         name=category,
                                         marker_color=color_dict[category],
                                         showlegend=False),
                            row=1,
                            col=2)


    # Get GRE stats
    gre_filt = (~u_df['grev'].isna()) & (~u_df['grem'].isna()) & (u_df['new_gre'])
    if sum(gre_filt) == 0:
        return pltly_fig

    dfq = u_df[gre_filt][['grem', hue]]
    dfq = dfq.assign(gre_type='Quant')
    dfq.columns = ['score', hue, 'gre_type']

    dfv = u_df[gre_filt][['grev', hue]]
    dfv = dfv.assign(gre_type='Verbal')
    dfv.columns = ['score', hue, 'gre_type']

    dfw = u_df[gre_filt][['grew', hue]]
    dfw = dfw.assign(gre_type='AWA')
    dfw.columns = ['score', hue, 'gre_type']

    cdf = pd.concat([dfq, dfv])

    for category in sorted(cdf[hue].unique().tolist()):
        pltly_fig.add_trace(go.Box(y=cdf[cdf[hue] == category]['score'],
                                   x=cdf[cdf[hue] == category]['gre_type'],
                                   name=category,
                                   marker_color=color_dict[category],
                                   showlegend=False,
                                   offsetgroup="vq" + category,
                                   alignmentgroup="vq"),
                            row=2,
                            col=1)

    for category in sorted(dfw[hue].unique().tolist()):
        pltly_fig.add_trace(go.Box(y=dfw[dfw[hue] == category]['score'],
                                   x=dfw[dfw[hue] == category]['gre_type'],
                                   name=category,
                                   marker_color=color_dict[category],
                                   showlegend=False,
                                   offsetgroup="aw" + category,
                                   alignmentgroup="aw"),
                            row=2,
                            col=2)

    inst_sep = ' - ' if len(field) > 0 else ''
    field_sep = ' - ' if degree is not None and len(degree) > 0 else ''
    if title is not None and len(title) > 20:
        if inst_sep != '':
            inst_sep = inst_sep + '<br>'
        elif field_sep != '':
            field_sep = field_sep + '<br>'

    pltly_fig.update_xaxes(showgrid=grid_lines, zeroline=False, showline=axis_lines, linecolor="#000")
    pltly_fig.update_xaxes(tickformat='%d %b', row=1, col=1)
    pltly_fig.update_yaxes(showgrid=grid_lines, zeroline=False, showline=axis_lines, linecolor="#000")
    pltly_fig.layout.title.text = f"{title if title is not None else 'All schools'}{inst_sep}{', '.join(field)}{field_sep}{', '.join(degree)}"
    pltly_fig.layout.title.font.size = 30
    pltly_fig.update_layout(
        title_x=0.5,
        margin=dict(t=120),
        autosize=False,
        height=900,
        paper_bgcolor="White",
        boxmode='group',
        barmode=gpa_multiple
    )

    pltly_fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        itemsizing="constant"
    ))

    return pltly_fig

@st.cache
def load_data():
    grad_df = pd.read_csv('app/data/full_data_clean.csv', index_col=0, low_memory=False)
    grad_df.loc[:, 'institution'] = grad_df['clean_institution'].str.strip()
    grad_df.loc[:, 'major'] = grad_df['clean_major'].str.strip()
    grad_df = grad_df[grad_df['gpafin'] <= 4]
    grad_df = grad_df[(grad_df['new_gre'] == True) | (grad_df['new_gre'].isna())]
    grad_df = grad_df[~grad_df['decdate'].isna()]
    grad_df.loc[:,'year'] = grad_df['decdate'].str[-4:].astype(int)
    grad_df = grad_df[(grad_df['year'] >= 2011) & (grad_df['year'] < datetime.datetime.now().year)]
    grad_df = grad_df[grad_df['season'] > 'F11']
    grad_df['decdate'] = pd.to_datetime(grad_df['decdate'], dayfirst=True)
    # Normalize to 2020. 2020 is a good choice because it's recent AND it's a leap year
    grad_df.loc[:, 'uniform_dates'] = vec_dt_replace(pd.to_datetime(grad_df['decdate']), year=2020)
    # Get december dates to be from "2019" so Fall decisions that came in Dec come before the Jan ones.
    dec_filter = grad_df['uniform_dates'] > datetime.datetime.strptime('2020-11-30', '%Y-%m-%d')
    grad_df.loc[dec_filter, 'uniform_dates'] = vec_dt_replace(pd.to_datetime(grad_df[dec_filter]['uniform_dates']), year=2019)
    grad_df.drop(columns=['comment',
                          'date_add',
                          'decdate',
                          'clean_institution',
                          'clean_major',
                          'date_add_ts',
                          'sub',
                          'decdate_ts',
                          'method'], inplace=True)
    grad_df['institution'] = grad_df['institution'].astype('category')
    grad_df['major'] = grad_df['major'].astype('category')
    grad_df['degree'] = grad_df['degree'].astype('category')
    grad_df['season'] = grad_df['season'].astype('category')
    grad_df['decisionfin'] = grad_df['decisionfin'].astype('category')
    grad_df['status'] = grad_df['status'].astype('category')
    grad_df['gpafin'] = grad_df['gpafin'].astype(np.float16)
    grad_df['grev'] = grad_df['grev'].astype(np.float16)
    grad_df['grem'] = grad_df['grem'].astype(np.float16)
    grad_df['grew'] = grad_df['grew'].astype(np.float16)
    grad_df['new_gre'] = grad_df['new_gre'].astype('bool')
    grad_df['year'] = grad_df['year'].astype(np.int16)
    logger.info(grad_df.memory_usage(deep=True) / 1024 ** 2)
    return grad_df

logger = log_to_file()
grad_df = load_data()

insts = grad_df['institution'].drop_duplicates().sort_values().tolist()
insts.insert(0, None)
majors = grad_df['major'].drop_duplicates().sort_values().tolist()
degrees = grad_df['degree'].drop_duplicates().sort_values().tolist()
seasons = grad_df['season'].drop_duplicates().sort_values().tolist()
status = grad_df['status'].drop_duplicates().sort_values().tolist()

st.title('GradCafe Stats Generator')
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.block-container {
    padding: 0px;
}
.main {
    padding-left: 3em;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.sidebar.markdown('## Filters')

# Institution filter
inst_choice = st.sidebar.selectbox('Institution:', insts)
# Major filter
major_choice = st.sidebar.multiselect('Major:', majors)
# Degree filter
deg_choice = st.sidebar.multiselect('Degree:', degrees)
# Season filter
season_choice = st.sidebar.multiselect('Season:', seasons)
# Status filter
status_choice = st.sidebar.multiselect('Status:', status)

st.sidebar.markdown('## Display options')

# Display options
stack = st.sidebar.checkbox('Stack bars in GPA plot', value=False)
axis_lines = st.sidebar.checkbox('Show axis lines in plots', value=False)
grid_lines = st.sidebar.checkbox('Show grid lines in plots', value=True)

process = psutil.Process(os.getpid())
logger.info(f"{inst_choice}|{major_choice}|{deg_choice}|{season_choice}|{status_choice}|{process.memory_info().rss / 1024 **2}")

pltly_fig = get_uni_stats(grad_df,
              search=inst_choice,
              title=inst_choice,
              degree=deg_choice,
              field=major_choice,
              status=status_choice,
              seasons=season_choice,
              stack=stack,
              axis_lines=axis_lines,
              grid_lines=grid_lines,
              debug=False)

st.plotly_chart(pltly_fig, use_container_width=True)
gc.collect()

