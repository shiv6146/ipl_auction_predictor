import joblib
import streamlit as st
import pandas as pd
import altair as alt
from model.crawlers import utils
from babel.numbers import format_currency
from datetime import datetime
from urllib.parse import urlparse

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

# @st.cache(persist=True)
@st.cache_data
def get_data(file_name):
    df = pd.read_csv(file_name)
    return df

# @st.cache(persist=True, allow_output_mutation=True)
@st.cache_resource
def load_models():
    reg = joblib.load('./model/auto_reg_v1.joblib')
    clf = joblib.load('./model/auto_clf_v1.joblib')
    return reg, clf

def is_valid_url(url: str) -> bool:
    """
    Validate a URL by checking if it has a scheme and a netloc.
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    processed_df = None
    # Drop unwanted columns
    processed_df = df.drop(['height', 'bat_style', 'bowl_style', 't20_no', 't20_avg', 't20_50', 
    't20_4s', 'ipl_no', 'ipl_avg', 'ipl_50', 'ipl_4s', 't20_bowl_avg', 'ipl_bowl_avg'], axis=1)
    # convert all stats column to numeric
    processed_df = processed_df.replace('-', 0.0)
    for col in processed_df.columns[4:16]:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    # merge columns together
    processed_df['total_runs'] = processed_df['t20_runs'] + processed_df['ipl_runs']
    processed_df['total_6s'] = processed_df['t20_6s'] + processed_df['ipl_6s']
    processed_df['total_sr'] = processed_df['t20_sr'] / 2 + processed_df['ipl_sr'] / 2
    processed_df['total_wkts'] = processed_df['t20_wkts'] + processed_df['ipl_wkts']
    processed_df['total_bowl_econ'] = processed_df['t20_bowl_econ'] / 2 + processed_df['ipl_bowl_econ'] / 2
    processed_df['total_bowl_sr'] = processed_df['t20_bowl_sr'] / 2 + processed_df['ipl_bowl_sr'] / 2

    # drop all merged cols
    processed_df = processed_df.drop(['t20_runs', 'ipl_runs', 't20_6s', 'ipl_6s', 'ipl_sr', 
    't20_sr', 'ipl_wkts', 't20_wkts', 'ipl_bowl_econ', 't20_bowl_econ', 't20_bowl_sr', 'ipl_bowl_sr'], axis=1)
    
    return processed_df

def show_price_spent_vs_six_hitting_ability_plot(df):
    tdf = pd.DataFrame({
        'sixes_count':df.groupby(["year", "team"])['total_6s'].sum(),
        'total_price': df.groupby(["year", "team"])['price'].sum()
        }).reset_index()
    
    tdf['total_price'] = tdf['total_price'] / 10000000


    c = alt.Chart(tdf).mark_trail().encode(
        x=alt.X('year', bin = False, scale=alt.Scale(domain=[2013, 2025]), axis=alt.Axis(format='d')), y='sixes_count', size='total_price', color='team', tooltip=['team', 'year', 'total_price', 'sixes_count']
        )
    
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='multi', nearest=True, on='mouseover',
                            fields=['year', 'sixes_count', 'total_price', 'team'], empty=False)
    
    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(tdf).mark_point().encode(
        x='year',
        y='sixes_count',
        size='total_price',
        color='team',
        opacity=alt.value(0),
    ).add_params(
        nearest
    )

    # Draw points on the line, and highlight based on selection
    points = c.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw a rule at the location of the selection
    rules = alt.Chart(tdf).mark_rule(color='gray').encode(
        x='year',
    ).transform_filter(
        nearest
    )

    # Put the five layers into a chart and bind the data
    final = alt.layer(
        c, selectors, points, rules
    ).properties(
        title="Six hitting ability of teams (vs) amount spent"
    ).interactive()

    st.altair_chart(final, use_container_width=True)

def show_price_spent_vs_crucial_roles_plot(df):
    tdf = pd.DataFrame({
        'role_count':df.groupby(["year", "role"])['role'].count(),
        'total_price': df.groupby(["year", "role"])['price'].sum()
        }).reset_index()
    
    tdf['total_price'] = tdf['total_price'] / 10000000

    c = alt.Chart(tdf).mark_trail().encode(
        x=alt.X('year', bin = False, scale=alt.Scale(domain=[2013, 2025]), axis=alt.Axis(format='d')), y='total_price', size='role_count', color='role', 
        tooltip=['year', 'role', 'role_count', 'total_price']
        )
    
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='multi', nearest=True, on='mouseover',
                            fields=['year', 'role', 'role_count', 'total_price'], empty=False)
    
    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(tdf).mark_point().encode(
        x='year',
        y='total_price',
        color='role',
        size='role_count',
        opacity=alt.value(0),
    ).add_params(
        nearest
    )

    # Draw points on the line, and highlight based on selection
    points = c.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw a rule at the location of the selection
    rules = alt.Chart(tdf).mark_rule(color='gray').encode(
        x='year',
    ).transform_filter(
        nearest
    )

    # Put the five layers into a chart and bind the data
    final = alt.layer(
        c, selectors, points, rules
    ).properties(
        title="Crucial roles (vs) amount spent"
    ).interactive()

    st.altair_chart(final, use_container_width=True)

def show_avg_age_per_team_over_the_years_plot(df):
    tdf = df.copy()
    # update age col offset vs curr year and curr age
    # tdf['age'] = tdf['age'] - (2025 - tdf['year'])
    tdf = pd.DataFrame({
        'avg_age': df.groupby(["year", "team"])['age'].mean()
        }).reset_index()

    c = alt.Chart(tdf).mark_trail().encode(
        x=alt.X('year', bin = False, scale=alt.Scale(domain=[2013, 2025]), axis=alt.Axis(format='d')), 
        y=alt.Y('avg_age', bin = False, scale=alt.Scale(domain=[20, 35]), axis=alt.Axis(format='d')), color='team', 
        tooltip=['year', 'team', 'avg_age']
        )
    
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='multi', nearest=True, on='mouseover',
                            fields=['year', 'team', 'avg_age'], empty=False)
    
    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(tdf).mark_point().encode(
        x='year',
        y='avg_age',
        color='team',
        opacity=alt.value(0),
    ).add_params(
        nearest
    )

    # Draw points on the line, and highlight based on selection
    points = c.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw a rule at the location of the selection
    rules = alt.Chart(tdf).mark_rule(color='gray').encode(
        x='year',
    ).transform_filter(
        nearest
    )

    # Put the five layers into a chart and bind the data
    final = alt.layer(
        c, selectors, points, rules
    ).properties(
        title="Average age of players per team targetted in every auction"
    ).interactive()

    st.altair_chart(final, use_container_width=True)

def show_top_batsr_price_over_the_years_plot(df):
    tdf = df.copy()
    tdf['price'] = tdf['price'] / 10000000
    bat = tdf[tdf.role.isin(['bowling-allrounder', 'batting-allrounder', 'batsman', 'wk-batsman'])][tdf['total_runs'] >= 500]
    top_5_df = bat.drop_duplicates('name').nlargest(5, 'total_sr')
    players = st.multiselect(
        "Choose batsman / all-rounder", list(bat['name'].unique()), top_5_df['name'].to_list()
    )
    if not players:
        st.error("Please select at least one batsman / all-rounder")
    else:
        data = bat[bat.name.isin(players)]
        c = alt.Chart(data).mark_circle().encode(
            x=alt.X('year', bin = False, scale=alt.Scale(domain=[2013, 2025]), axis=alt.Axis(format='d')), 
            y='price', 
            color='name',
            size='total_sr',
            # size=alt.Size('total_sr', scale=alt.Scale(domain=[100, 300])),
            tooltip=['name', 'year', 'price', 'total_sr']
        )
        
        # Create a selection that chooses the nearest point & selects based on x-value
        nearest = alt.selection(type='multi', nearest=True, on='mouseover',
                                fields=['name', 'year', 'price', 'total_sr'], empty=False)
        
        # Transparent selectors across the chart. This is what tells us
        # the x-value of the cursor
        selectors = alt.Chart(data).mark_point().encode(
            x='year',
            y='price',
            color='name',
            opacity=alt.value(0),
        ).add_params(
            nearest
        )

        # Draw points on the line, and highlight based on selection
        points = c.mark_point().encode(
            opacity=alt.condition(nearest, alt.value(1), alt.value(0))
        )

        # Draw a rule at the location of the selection
        rules = alt.Chart(data).mark_rule(color='gray').encode(
            x='year',
        ).transform_filter(
            nearest
        )

        # Put the five layers into a chart and bind the data
        final = alt.layer(
            c, selectors, points, rules
        ).properties(
            title="Price of players with best career batting strike rate"
        ).interactive()

        st.altair_chart(final, use_container_width=True)

def show_top_bowlsr_price_over_the_years_plot(df):
    tdf = df.copy()
    tdf['price'] = tdf['price'] / 10000000
    bowl = tdf[tdf.role.isin(['bowling-allrounder', 'bowler'])][tdf['total_wkts'] >= 100]
    top_5_df = bowl.drop_duplicates('name').nsmallest(5, 'total_bowl_econ')
    players = st.multiselect(
        "Choose bowler / all-rounder", list(bowl['name'].unique()), top_5_df['name'].to_list()
    )
    if not players:
        st.error("Please select at least one bowler / all-rounder")
    else:
        data = bowl[bowl.name.isin(players)]
        c = alt.Chart(data).mark_circle().encode(
            x=alt.X('year', bin = False, scale=alt.Scale(domain=[2013, 2025]), axis=alt.Axis(format='d')), 
            y='price', 
            color='name',
            size='total_bowl_econ',
            # size=alt.Size('total_sr', scale=alt.Scale(domain=[100, 300])),
            tooltip=['name', 'year', 'price', 'total_bowl_econ']
        )
        
        # Create a selection that chooses the nearest point & selects based on x-value
        nearest = alt.selection(type='multi', nearest=True, on='mouseover',
                                fields=['name', 'year', 'price', 'total_bowl_econ'], empty=False)
        
        # Transparent selectors across the chart. This is what tells us
        # the x-value of the cursor
        selectors = alt.Chart(data).mark_point().encode(
            x='year',
            y='price',
            color='name',
            opacity=alt.value(0),
        ).add_params(
            nearest
        )

        # Draw points on the line, and highlight based on selection
        points = c.mark_point().encode(
            opacity=alt.condition(nearest, alt.value(1), alt.value(0))
        )

        # Draw a rule at the location of the selection
        rules = alt.Chart(data).mark_rule(color='gray').encode(
            x='year',
        ).transform_filter(
            nearest
        )

        # Put the five layers into a chart and bind the data
        final = alt.layer(
            c, selectors, points, rules
        ).properties(
            title="Price of players with best career bowling economy"
        ).interactive()

        st.altair_chart(final, use_container_width=True)

try:
    df = preprocess(get_data('./data/data.csv'))
    reg_model, clf_model = load_models()

    st.title('IPL Auction Prediction')
    col1, col2 = st.columns(2)
    with col1:
        player_url = st.text_input('Paste Cricbuzz profile link of any player')
        auction_yr = st.number_input('Year of auction', datetime.now().year, datetime.now().year+2)
        
    with col2:
        player = None
        with st.spinner(text='Fetching player info...'):
            player_feat, player = utils.get_player_features(player_url, auction_yr)
            if player_url != '' and player is None:
                st.error(f'Oops! Could not fetch player')
        with st.spinner(text=f'Making predictions for player'):
            if player_url != '' and player is not None:
                predicted_price = format_currency(reg_model.predict(player_feat)[0] / 10000000, 'INR', locale='en_IN', format=u"₹#,##0.00 'Cr'")
                predicted_team = clf_model.predict(player_feat)[0].replace('-', ' ').title()
                player_copy = player_feat.copy()
                player_copy['name'] = player.name.title()
                st.write('Player Stats Summary')
                st.write(player_copy.set_index('name'))
                st.success(f'**{predicted_team}** could place a bid of **{predicted_price}** for **{player.name.title()}** in the **{auction_yr}** IPL auction')

    st.title('IPL Auction Data Analysis [2013 - 2025](https://www.iplt20.com/auction/2025)')
    show_price_spent_vs_six_hitting_ability_plot(df)
    col3, col4 = st.columns(2)
    with col3:
        show_price_spent_vs_crucial_roles_plot(df)
        show_top_batsr_price_over_the_years_plot(df)
    with col4:
        show_avg_age_per_team_over_the_years_plot(df)
        show_top_bowlsr_price_over_the_years_plot(df)
except Exception as e:
    st.error(
        """
        **Something went wrong**
        Error: %s
    """
        % e
    )