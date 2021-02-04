import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import time
import datetime
import logging
import os, psutil
from typing import Union

st.set_page_config(page_title='Grad Stats',
         page_icon=':coffee:')

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
    title = title if title is not None else search
    if isinstance(degree, str) and degree not in ['MS', 'PhD', 'MEng', 'MFA', 'MBA', 'Other']:
        degree = 'PhD'

    if stack:
        gpa_multiple = 'stack'
    else:
        gpa_multiple = 'dodge'

    # Trying to pick red/green colorblind-friendly colors
    if grid_lines:
        sns.set_theme(style="whitegrid", font_scale=1)
    else:
        sns.set_theme(style="white", font_scale=1)
    flatui = ["#2eff71", "#ff0000", "#0000ff"]
    sns.set_palette(flatui)
    acc_patch = mpatches.Patch(color='#2eff7180')
    rej_patch = mpatches.Patch(color='#ff000080')
    int_patch = mpatches.Patch(color='#0000ff80')
    acc_line = mlines.Line2D([], [], color='#2eff71')
    rej_line = mlines.Line2D([], [], color='#ff0000')
    int_line = mlines.Line2D([], [], color='#0000ff')
  
    hue_order = ['Accepted', 'Rejected', 'Interview']
    if hue == 'status':
        hue_order = ['American', 'International', 'International with US Degree', 'Other']

    if debug:
        # We're logging at an info level because of streamlits logger...
        logger.info("prep time: %.2f seconds" % (time.time() - start_time))
        start_time = time.time()

    # This generates 4 graphs, so let's make it a 2x2 grid
    fig, ax = plt.subplots(2,2, dpi=80)
    fig.set_size_inches(10, 10)
    
    # Timeline stats
    u_df = u_df[create_filter(u_df, degree, ['Accepted', 'Rejected', 'Interview'], search, field, status, seasons)]
    if len(u_df) < 1:
        return fig

    if debug:
        logger.info("filter time: %.2f seconds" % (time.time() - start_time))
        start_time = time.time()

    graph_filt =  u_df['uniform_dates'] < datetime.datetime(2020, 6, 1)
    if sum(graph_filt) == 0:
        return fig

    if debug:
        logger.info("preptimeline time: %.2f seconds" % (time.time() - start_time))
        start_time = time.time()

    sns.histplot(data=u_df[graph_filt],
                 x='uniform_dates',
                 hue=hue,
                 cumulative=True,
                 discrete=False,
                 element='step',
                 fill=False,
                 hue_order=hue_order,
                 ax=ax[0][0])

    if debug:
        logger.info("actual timeline plot time: %.2f seconds" % (time.time() - start_time))
        start_time = time.time()

    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    formatter.formats = ['%b',  # years
                         '%b',       # months
                         '%d',       # days
                         '%H:%M',    # hrs
                         '%H:%M',    # min
                         '%S.%f', ]  # secs
    # Hide the year
    formatter.zero_formats = ['%b',  # years
                         '%b',       # months
                         '%d',       # days
                         '%H:%M',    # hrs
                         '%H:%M',    # min
                         '%S.%f', ]  # secs
    # Hide the year
    formatter.offset_formats = ['',  # years
                         '',       # months
                         '%d',       # days
                         '%H:%M',    # hrs
                         '%H:%M',    # mins
                         '%S.%f', ]  # secs
    ax[0][0].xaxis.set_major_locator(locator)
    ax[0][0].xaxis.set_major_formatter(formatter)
    h, l = ax[0][0].get_legend_handles_labels()
    # Add frequency counts
    if h is not None and l is not None:
        if hue == 'decisionfin':
            counts = u_df[graph_filt][hue].value_counts().reindex(hue_order)
            l = [f'{value} (n={count})' for value, count in counts.iteritems()]
            ax[0][0].legend(handles=[acc_line, rej_line, int_line], labels=l, title="Decision", fontsize=8)

    ax[0][0].set_xlabel("Date")
    ax[0][0].set_ylabel("Count")
    ax[0][0].set_title("Decision Timeline", fontsize=15)
    if debug:
        logger.info("timeline aesthetics time: %.2f seconds" % (time.time() - start_time))
        start_time = time.time()
    
    # Get GPA stats
    gpa_filt = (~u_df['gpafin'].isna()) & (u_df['gpafin'] <= 4)
    gpa_bins = 10 if gpa_multiple == 'dodge' else 20
    if sum(gpa_filt) > 0:
        sns.histplot(data=u_df[gpa_filt],
                     x='gpafin',
                     hue=hue,
                     hue_order=hue_order,
                     multiple=gpa_multiple,
                     bins=gpa_bins,
                     ax=ax[0][1])
        ax[0][1].set_xlabel("GPA")
        ax[0][1].set_ylabel("Count")
        ax[0][1].set_title("GPA Distribution ({0} values)".format(gpa_multiple), fontsize=15)
        # Add frequency counts
        h, l = ax[0][1].get_legend_handles_labels()
        if h is not None and l is not None:
            if hue == 'decisionfin':
                counts = u_df[gpa_filt][hue].value_counts().reindex(hue_order)
                l = [f'{value} (n={count})' for value, count in counts.iteritems()]
                ax[0][1].legend(handles=[acc_patch, rej_patch, int_patch], labels=l, title="Decision", fontsize=8)
    if debug:
        logger.info("gpa time: %.2f seconds" % (time.time() - start_time))
        start_time = time.time()

    # Get GRE stats
    gre_filt = (~u_df['grev'].isna()) & (~u_df['grem'].isna()) & (u_df['new_gre'])
    if sum(gre_filt) == 0:
        return fig
    dfq = u_df[gre_filt][['grem', hue]]
    dfq = dfq.assign(gre_type='Quant')
    dfq.columns = ['score', hue, 'gre_type']

    dfv = u_df[gre_filt][['grev', hue]]
    dfv = dfv.assign(gre_type='Verbal')
    dfv.columns = ['score', hue, 'gre_type']

    cdf = pd.concat([dfq, dfv])
    sns.boxplot(data=cdf,
                x='gre_type',
                y='score',
                hue=hue,
                linewidth=1,
                fliersize=1,
                hue_order=hue_order,
                ax=ax[1][0])
    leg = ax[1][0].get_legend()
    if leg is not None:
        leg.set_title('Decision')
    ax[1][0].set_xlabel("GRE Section")
    ax[1][0].set_ylabel("Score")
    ax[1][0].set_title("GRE Score distribution", fontsize=15)
    if debug:
        logger.info("greqv time: %.2f seconds" % (time.time() - start_time))
        start_time = time.time()
    
    # Get GRE AWA stats
    gre_filt = (~u_df['grew'].isna()) & (u_df['new_gre'])
    sns.boxplot(data=u_df[gre_filt],
                x=['AWA'] * len(u_df[gre_filt]),
                y='grew',
                hue=hue,
                linewidth=1,
                fliersize=1,
                hue_order=hue_order,
                ax=ax[1][1])
    leg = ax[1][1].get_legend()
    if leg is not None:
        leg.set_title('Decision')
    ax[1][1].set_xlabel("GRE Section")
    ax[1][1].set_ylabel("Score")
    ax[1][1].set_title("GRE AWA Score distribution", fontsize=15)
    if debug:
        logger.info("grew time: %.2f seconds" % (time.time() - start_time))
        start_time = time.time()
    

    if not axis_lines:
        sns.despine(left=True)
    inst_sep = ' - ' if len(field) > 0 else ''
    field_sep = ' - ' if degree is not None and len(degree) > 0 else ''
    fig.suptitle(f"{title if title is not None else 'All schools'}{inst_sep}{', '.join(field)}{field_sep}{', '.join(degree)}", size=25)
    fig.tight_layout()
    return fig

@st.cache
def load_data():
    grad_df = pd.read_csv('app/data/full_data_clean.csv', index_col=0, low_memory=False)
    grad_df.loc[:, 'institution'] = grad_df['clean_institution'].str.strip()
    grad_df.loc[:, 'major'] = grad_df['clean_major'].str.strip()
    grad_df = grad_df[(grad_df['new_gre'] == True) | (grad_df['new_gre'].isna())]
    grad_df = grad_df[~grad_df['decdate'].isna()]
    grad_df.loc[:,'year'] = grad_df['decdate'].str[-4:].astype(int)
    grad_df = grad_df[(grad_df['year'] >= 2011) & (grad_df['year'] < datetime.datetime.now().year)]
    grad_df = grad_df[grad_df['season'] > 'F11']
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
    grad_df['year'] = grad_df['year'].astype(np.int8)
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
.block-container {padding: 0px}
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
grid_lines = st.sidebar.checkbox('Show grid lines in plots', value=False)

process = psutil.Process(os.getpid())
logger.info(f"{inst_choice},{major_choice},{deg_choice},{season_choice},{status_choice},{process.memory_info().rss / 1024 **2}")
fig = get_uni_stats(grad_df,
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

st.pyplot(fig)
