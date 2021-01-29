import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import datetime
from typing import Union

def vec_dt_replace(series, year=None, month=None, day=None):
    return pd.to_datetime(
        {'year': series.dt.year if year is None else year,
         'month': series.dt.month if month is None else month,
         'day': series.dt.day if day is None else day})

@st.cache
def create_filter(df,
                  degree: str = None,
                  decisionfin: Union[str, list] = None,
                  institution: Union[str, list] = None,
                  field: Union[str, list] = None,
                  status: Union[str, list] = None,
                  seasons: Union[str, list] = None,
                  gpa: bool = False,
                  gre: bool = False):
    filt = [True] * len(df)

    # NEEDS REFACTORING
    if degree is not None and len(degree) > 0:
        if isinstance(degree, str):
            filt = (filt) & (df['degree'].str.contains(degree, case=False))
        elif isinstance(degree, list):
            filt = (filt) & (df['degree'].isin(degree))
    if decisionfin is not None:
        if isinstance(decisionfin, str):
            filt = (filt) & (df['decisionfin'].str.contains(decisionfin, case=False))
        elif isinstance(decisionfin, list):
            filt = (filt) & (df['decisionfin'].isin(decisionfin))
    if institution is not None:
        if isinstance(institution, str):
            filt = (filt) & (df['institution'].str.contains(institution, case=False))
        elif isinstance(institution, list):
            filt = (filt) & (df['institution'].isin(institution))
    if field is not None and len(field) > 0:
        if isinstance(field, str):
            filt = (filt) & (df['major'] == field)
        elif isinstance(field, list):
            filt = (filt) & (df['major'].isin(field))
    if status is not None and len(status) > 0:
        if isinstance(status, str):
            filt = (filt) & (df['status'] == status)
        elif isinstance(status, list):
            filt = (filt) & (df['status'].isin(status))
    if seasons is not None and len(seasons) > 0:
        if isinstance(seasons, str):
            filt = (filt) & (df['season'] == seasons)
        elif isinstance(seasons, list):
            filt = (filt) & (df['season'].isin(seasons))
    if gpa:
        filt = (filt) & (~df['gpafin'].isna()) & (df['gpafin'] <= 4)
    if gre:
        filt = (filt) & (~df['grev'].isna()) & (~df['grem'].isna()) & (~df['grew'].isna()) & (df['new_gre'])
    
    return filt

def get_uni_stats(u_df,
					search: str = None,
					title: str = None,
					degree: str = 'PhD',
					field: str = None,
					status: list = None,
					seasons: list = None,
                    stack: bool = False,
					hue: str = 'decisionfin',
                    filt = None):
    title = title if title is not None else search
    if isinstance(degree, str) and degree not in ['MS', 'PhD', 'MEng', 'MFA', 'MBA', 'Other']:
        degree = 'PhD'

    if stack:
        gpa_multiple = 'stack'
    else:
        gpa_multiple = 'dodge'
    # Clean up the data a bit, this probably needs a lot more work
    # Maybe its own method, too
    u_df = u_df.copy()
    u_df = u_df[~u_df['decdate'].isna()]
    u_df.loc[:,'year'] = u_df['decdate'].str[-4:].astype(int)
    u_df = u_df[(u_df['year'] > 2000) & (u_df['year'] < datetime.datetime.now().year)]
    # Normalize to 2020. 2020 is a good choice because it's recent AND it's a leap year
    u_df.loc[:, 'uniform_dates'] = vec_dt_replace(pd.to_datetime(u_df['decdate']), year=2020)
    # Get december dates to be from "2019" so Fall decisions that came in Dec come before the Jan ones.
    dec_filter = u_df['uniform_dates'] > datetime.datetime.strptime('2020-11-30', '%Y-%m-%d')
    u_df.loc[dec_filter, 'uniform_dates'] = vec_dt_replace(pd.to_datetime(u_df[dec_filter]['uniform_dates']), year=2019)

    # Trying to pick red/green colorblind-friendly colors
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

    # This generates 4 graphs, so let's make it a 2x2 grid
    fig, ax = plt.subplots(2,2)
    fig.set_size_inches(20, 20)
    
    # Timeline stats
    if filt is None:
        mscs_filt = create_filter(u_df, degree, ['Accepted', 'Rejected', 'Interview'], search, field, status, seasons)
    else:
        mscs_filt = filt
    mscs_filt = (mscs_filt) & (u_df['uniform_dates'].astype(str) <= '2020-06-00')
    if sum(mscs_filt) == 0:
        return fig
    sns.histplot(data=u_df[mscs_filt],
                 x='uniform_dates',
                 hue=hue,
                 cumulative=True,
                 discrete=False,
                 element='step',
                 fill=False,
                 hue_order=hue_order,
                 ax=ax[0][0])

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
            counts = u_df[mscs_filt][hue].value_counts().reindex(hue_order)
            l = [f'{value} (n={count})' for value, count in counts.iteritems()]
            ax[0][0].legend(handles=[acc_line, rej_line, int_line], labels=l, title="Decision")

    ax[0][0].set_xlabel("Date")
    ax[0][0].set_ylabel("Count")
    ax[0][0].set_title("Cumsum of decisions")
    
    # Get GPA stats
    gpa_filt = (mscs_filt) & (~u_df['gpafin'].isna()) & (u_df['gpafin'] <= 4)
    if sum(gpa_filt) > 0:
        sns.histplot(data=u_df[gpa_filt],
                     x='gpafin',
                     hue=hue,
                     hue_order=hue_order,
                     multiple=gpa_multiple,
                     bins=20,
                     ax=ax[0][1])
        ax[0][1].set_xlabel("GPA")
        ax[0][1].set_ylabel("Count")
        ax[0][1].set_title("GPA Distribution (stacked values)")
        # Add frequency counts
        h, l = ax[0][1].get_legend_handles_labels()
        if h is not None and l is not None:
            if hue == 'decisionfin':
                counts = u_df[gpa_filt][hue].value_counts().reindex(hue_order)
                l = [f'{value} (n={count})' for value, count in counts.iteritems()]
                ax[0][1].legend(handles=[acc_patch, rej_patch, int_patch], labels=l, title="Decision")

    # Get GRE stats
    gre_filt = (mscs_filt) & (~u_df['grev'].isna()) & (~u_df['grem'].isna()) & (u_df['new_gre'])
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
                linewidth=2.5,
                hue_order=hue_order,
                ax=ax[1][0])
    leg = ax[1][0].get_legend()
    if leg is not None:
        leg.set_title('Decision')
    ax[1][0].set_xlabel("GRE Section")
    ax[1][0].set_ylabel("Score")
    ax[1][0].set_title("GRE Score distribution")
    
    # Get GRE AWA stats
    gre_filt = (mscs_filt) & (~u_df['grew'].isna()) & (u_df['new_gre'])
    sns.boxplot(data=u_df[gre_filt],
                x=['AWA'] * len(u_df[gre_filt]),
                y='grew',
                hue=hue,
                linewidth=2.5,
                hue_order=hue_order,
                ax=ax[1][1])
    leg = ax[1][1].get_legend()
    if leg is not None:
        leg.set_title('Decision')
    ax[1][1].set_xlabel("GRE Section")
    ax[1][1].set_ylabel("Score")
    ax[1][1].set_title("GRE AWA Score distribution")
    
    # Save file to output directory
    fig.suptitle(title + ', ' + ', '.join(field) + ' ' + ', '.join(degree) + '', size='xx-large')
    return fig

@st.cache
def load_data():
    grad_df = pd.read_csv('app/data/full_data.csv', index_col=0, low_memory=False)
    grad_df.loc[:, 'institution'] = grad_df['institution'].str.strip()
    grad_df.loc[:, 'institution'] = grad_df['institution'].str.replace(r'[^\w\(\) ]', '', regex=True)
    grad_df.loc[:, 'major'] = grad_df['major'].str.strip()
    grad_df.loc[:, 'major'] = grad_df['major'].str.replace(r'[^\w ]\(\)', '', regex=True)
    grad_df = grad_df[(grad_df['new_gre'] == True) | (grad_df['new_gre'].isna())]
    return grad_df

grad_df = load_data()

insts = grad_df['institution'].drop_duplicates().sort_values().tolist()
majors = grad_df['major'].drop_duplicates().sort_values().tolist()
degrees = grad_df['degree'].drop_duplicates().sort_values().tolist()
seasons = grad_df['season'].drop_duplicates().sort_values().tolist()
status = grad_df['status'].drop_duplicates().sort_values().tolist()

st.title('GradCafe Stats Generator')

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

# grad_df.drop(columns=['decdate_ts', 'date_add_ts'], inplace=True)
# grad_df.columns = ['Institution',
# 					'Major',
# 					'Degree',
# 					'Season',
# 					'Decision',
# 					'Notification Method',
# 					'Decision Date',
# 					'GPA',
# 					'GRE Verbal',
# 					'GRE Quant',
# 					'GRE AWA',
# 					'GRE Subject',
# 					'Status',
# 					'Entry Date',
# 					'Comment']

fig = get_uni_stats(grad_df,
			  search=inst_choice,
			  title=inst_choice,
			  degree=deg_choice,
			  field=major_choice,
			  status=status_choice,
			  seasons=season_choice,
              stack=stack)

st.pyplot(fig)
