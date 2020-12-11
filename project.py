#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pylab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from wordcloud import WordCloud
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


# In[2]:


import os
if not os.path.exists('./figs'):
    os.makedirs('./figs')


# 2020 has been a tough year to the entire world. Coronavirus has become a global tragedy in the history. Many people are affected in the disaster in many ways. Up to early October, Coronavirus has spread to 218 countries around the world, and there are 44,591,343 confirmed cases and 13,253,657 deaths of Covid-19 reported globally. Besides the tremendous number of people who suffered from Covid-19, many others are forced to stay at home. Let's walk through some visualizations created from a few Covid-19 datasets to have a closer look at this disaster from various aspects. To start off, let's see a treemap of confirmed cases of Covid-19 across continents. From this graph, we can easily tell see that Americas, Asia, Europe, and Africa all have been affected. Moreover, we are able to pinpoint the mostly affected countries in each continent by the area of each square. It's very visually straight forward that the bigger the square, the more reported cases in that country. For example, in Americas, the top countries that have the most cases reported are the United States, Brazil, Argentina, and so on. In Asia, the top countries are India, Iran, Iraq, and so on. In Europe, the top countries are Russia, France, Spain, and etc. 

# In[43]:


df = pd.read_csv('global.csv')
df['continent'] = np.where((df.continent == 'North America'),'Americas',df.continent)
df['continent'] = np.where((df.continent == 'South America'),'Americas',df.continent)
df = df[(df['continent'] == 'Americas') | (df['continent'] == 'Asia') | (df['continent'] == 'Africa') | (df['continent'] == 'Europe')]
df = df[df['date'] == '10/29/20']
df["world"] = "world"
df = df.dropna(axis=0)
tree = df.sort_values(by=['continent', 'total_cases'], ascending=False)
color_pallate = ["slateblue", 'lightsteelblue', 'thistle', 'lightpink', 'indianred'][::-1]


# In[ ]:



fig = px.treemap(tree, path=['world', 'continent', 'location'], values='total_cases',
                  color='total_cases',
                  color_continuous_scale=color_pallate)  
fig.update_layout(
    title={
        'text': "Treemap for Confirmed Cases of Covid-19 WorldWide",
        'y':0.93,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font_size':18})
fig.show()
fig.write_image("figs/fig1.png")


# Among all of the countries that are affected, the top five countries with the most reported cases of Covid-19 are the United States, India, Brazil, Russia, and France. Now, let's take a closer look at these five countries by observing the area chart that the growth of reported cases of Covid-19 from January to October in each of the five countries. The general trend is that the coruntries start to have reported cases in March and the numbers growly gradually until June. From June to September, the reported confirmed cases spiked up. 

# In[3]:


df = pd.read_csv('global.csv')

def split_month(row):
    month = int(row['date'].split('/')[0])
    return month
def split_date(row):
    date = int(row['date'].split('/')[1])
    return date
df['month'] = df.apply(split_month, axis=1)
df['day'] = df.apply(split_date, axis=1)

new_df = df[df['day'] == 20]
area = new_df[(new_df['location'] == 'United States') | (new_df['location'] == 'India') | (new_df['location'] == 'Brazil') | 
             (new_df['location'] == 'Russia') | (new_df['location'] == 'France')]

area = area.fillna(0)
us = area[area['location'] == 'United States']['total_cases']
ind = area[area['location'] == 'India']['total_cases']
br = area[area['location'] == 'Brazil']['total_cases']
ra = area[area['location'] == 'Russia']['total_cases']
fr = area[area['location'] == 'France']['total_cases']

month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
fig = go.Figure()
fig.add_trace(go.Scatter(x=month, y=us, fill='tozeroy', line_color='cornflowerblue', name='U.S.')) 
fig.add_trace(go.Scatter(x=month, y=ind, fill='tozeroy', line_color='orange', name='India')) 
fig.add_trace(go.Scatter(x=month, y=br, fill='tozeroy', line_color='pink', name='Brazil')) 
fig.add_trace(go.Scatter(x=month, y=ra, fill='tozeroy', line_color='yellow', name='Russia')) 
fig.add_trace(go.Scatter(x=month, y=fr, fill='tozeroy', line_color='green', name='France')) 
fig.update_layout(hovermode='x unified')
fig.update_layout(
    title={
        'text': "Top 5 Countries with the Most Confirmed Cases of Covid-19 from January to October",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font_size': 16},
    xaxis_title="Months",
    yaxis_title="Population")
fig.show()
fig.write_image("figs/fig2.png")


# In[ ]:





# In[ ]:





# Let's now focus on the top 1 country with the most reported confirmed cases, the United States. In October, there are 82,147,55 reported confirmed cases and 198,253 deaths of Covid-19 in the United States. From violin plots below, we get a clear visualization of the distribution of confirmed cases, deaths, confirmed cases in the last 7 days, deaths in the last 7 days reported by all the states. All four violin plots are highly skewed which align with the fact that some states are indeed more severly affected than others. 

# In[81]:


map_df = pd.read_csv('covid_by_state.csv')
map_df = map_df.rename(columns={'Total_Cases': 'Total Cases'})
states = list(map_df['State_Abb'])
cases = list(map_df['Total Cases'])
death = list(map_df['Total Deaths'])
last_week_confirmed = list(map_df['Cases in Last 7 Days'])
last_week_death = list(map_df['Deaths in Last 7 Days'])

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 7))

plt.subplot(1, 4, 1)
v1 = plt.violinplot(cases, showmeans=False, showmedians=True)
v1['cmaxes'].set_edgecolor('#668536')
v1['cmins'].set_edgecolor('#668536')
v1['cbars'].set_edgecolor('#668536')
v1['cmedians'].set_edgecolor('#668536')
v1['bodies'][0].set_facecolor('#BAE676')
v1['bodies'][0].set_edgecolor('#668536')
v1['bodies'][0].set_linewidth(3)
v1['bodies'][0].set_alpha(0.5)
plt.title('Total Confirmed Cases', fontsize=14)
plt.xlabel('Total Confirmed Cases', fontsize=12)
plt.ylabel('Population')
plt.xticks([], [])

plt.subplot(1, 4, 2)
v2 = plt.violinplot(death, showmeans=False, showmedians=True)
v2['cmaxes'].set_edgecolor('#6490AB')
v2['cmins'].set_edgecolor('#6490AB')
v2['cbars'].set_edgecolor('#6490AB')
v2['cmedians'].set_edgecolor('#6490AB')
v2['bodies'][0].set_facecolor('#A0D1EF')
v2['bodies'][0].set_edgecolor('#6490AB')
v2['bodies'][0].set_linewidth(3)
v2['bodies'][0].set_alpha(0.5)
plt.title('Total Deaths', fontsize=14)
plt.xlabel('Total Deaths')
plt.ylabel('Population')
plt.xticks([], [])

plt.subplot(1, 4, 3)
v3 = plt.violinplot(last_week_confirmed, showmeans=False, showmedians=True)
v3['cmaxes'].set_edgecolor('#9F6EAB')
v3['cmins'].set_edgecolor('#9F6EAB')
v3['cbars'].set_edgecolor('#9F6EAB')
v3['cmedians'].set_edgecolor('#9F6EAB')
v3['bodies'][0].set_facecolor('#EDD6F3')
v3['bodies'][0].set_edgecolor('#9F6EAB')
v3['bodies'][0].set_linewidth(3)
v3['bodies'][0].set_alpha(0.5)
plt.title('Confirmed Cases in Last 7 Days', fontsize=14)
plt.xlabel('Confirmed Cases in Last 7 Days')
plt.ylabel('Population')
plt.xticks([], [])

plt.subplot(1, 4, 4)
v4 = plt.violinplot(last_week_death, showmeans=False, showmedians=True)
v4['cmaxes'].set_edgecolor('#EE9605')
v4['cmins'].set_edgecolor('#EE9605')
v4['cbars'].set_edgecolor('#EE9605')
v4['cmedians'].set_edgecolor('#EE9605')
v4['bodies'][0].set_facecolor('#F4C06A')
v4['bodies'][0].set_edgecolor('#EE9605')
v4['bodies'][0].set_linewidth(3)
v4['bodies'][0].set_alpha(0.5)
plt.title('Deaths in Last 7 Days', fontsize=14)
plt.xlabel('Deaths in Last 7 Days')
plt.ylabel('Population')
plt.xticks([], [])

fig.subplots_adjust(wspace=3)
fig.suptitle('Affected Populatinon In the U.S. During Covid-19', fontsize=20)

plt.show()
fig.savefig("figs/fig3.png")


# So which states suffer more from Covid-19? We are interested in the spread of Covid-19 across states in America. From the choropleth of confirmed cases within the U.S., we see that all states are affected, and it's very noticeable that California, Texas, Floria, New York, and Illinois have the most confirmed cases because these states are represented by deeper colors compared to the others. 

# In[31]:


map_df = pd.read_csv('covid_by_state.csv')
map_df = map_df.rename(columns={'Total_Cases': 'Total Cases'})
states = list(map_df['State_Abb'])
cases = list(map_df['Total Cases'])
death = list(map_df['Total Deaths'])
last_week_confirmed = list(map_df['Cases in Last 7 Days'])
last_week_death = list(map_df['Deaths in Last 7 Days'])

fig = px.choropleth(locations=states, locationmode="USA-states", color=cases, scope="usa",
                   color_continuous_scale = "Reds",
                   labels={'locations': 'State', 'color': 'Total Cases'})
fig.update_layout(
    title={
        'text': "Confirmed Cases of Covid-19 in the United States by State",
        'y':0.93,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()
fig.write_image("figs/fig4.png")


# In[ ]:





# Let's focus on the top 10 states of the most confirmed cases, and they are California, Texas, Florida, New York, New Jersey, Arizona, Tennessee, Illinois, North Carolina, and Georgia. By observing the growth of confirmed cases in each of theses states from January to early october, the general trend is that the confirmed cases grow relatviely slow from January to May, and then the numbers start to increase rapidly from June. The growth the confirmed cases in the top 10 states are also consistent with the global trend we mentioned earlier. 

# In[32]:


df = pd.read_csv('United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv')

def split_month(row):
    month = int(row['submission_date'].split('/')[0])
    return month
def split_date(row):
    date = int(row['submission_date'].split('/')[1])
    return date
df['month'] = df.apply(split_month, axis=1)
df['date'] = df.apply(split_date, axis=1)
df = df[['submission_date', 'month', 'date','state', 'tot_cases', 'tot_death']]

line_df = df[(df['state']=='TX') | (df['state']=='CA') | (df['state']=='FL') | (df['state']=='NJ') | (df['state']=='IL')
             | (df['state']=='GA') | (df['state']=='NC') | (df['state']=='TN') | (df['state']=='AZ') | (df['state']=='NY')]
fig = px.line(line_df, x='submission_date', y="tot_cases", color='state',labels={"state": "State", "tot_cases": "Total Cases", "submission_date": "Date"}, 
                title="Total Cases of Covid19 in Top 10 States")
fig.update_layout(
    title={
        'text': "Total Cases of Covid-19 in Top 10 States",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_xaxes(
    ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'],
    tickvals=["01/25/2020", "02/25/2020", "03/25/2020", "04/25/2020", "05/25/2020", "06/25/2020", "07/25/2020", 
               "08/25/2020", "09/25/2020", "10/25/2020"])
fig.update_layout(hovermode='x unified')
fig.update_traces(line=dict(width=4))
fig.show()
fig.write_image("figs/fig5.png")


# In[ ]:





# We care about fatality as well. From another choropleth of total deaths within the U.S., we see that California, Texas, Floria, New Jersey, New York, and Illinois have the most number of deaths because these states are represented by deeper colors compared to the others. 

# In[33]:


fig = px.choropleth(locations=states, locationmode="USA-states", color=death, scope="usa",
                   color_continuous_scale = "Blues",
                   labels={'locations': 'State', 'color': 'Total Deaths'})
fig.update_layout(
    title={
        'text': "Total Deaths of Covid-19 in the United States by State",
        'y':0.93,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.show()
fig.write_image("figs/fig6.png")


# 

# We are able to pinpoint the top states with the most confirmed cases and deaths from the two choropleth maps, now let's have a more strightforward visualization to the top 10 states with the most confirmed cases reported and their reported deaths correspondingly. It brings to our attention that Arizona, New York, and New Jersey have the fewest confirmed cases among the top 10 states, however, New York and New Jersey have more deaths than even North Carolina and Tennessee, which reported more confirmed cases than these two states. This actually makes sense because some people recover more easily than others due to their health condition, immune system, and etc. For example, statistics show that elderly people have a higher fatality rate than young people when they have infection of Covid-19. So there are factors that could cause a state with less confirmed cases turn out having higher fatality. 

# In[86]:


df = pd.read_csv('United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv')

def split_month(row):
    month = int(row['submission_date'].split('/')[0])
    return month
def split_date(row):
    date = int(row['submission_date'].split('/')[1])
    return date
df['month'] = df.apply(split_month, axis=1)
df['date'] = df.apply(split_date, axis=1)
df = df[['submission_date', 'month', 'date','state', 'tot_cases', 'tot_death']]


bar_df = df[(df['date']==30) & (df['month']==10)]
bar_df = bar_df.sort_values('tot_cases', ascending=False)
bar_df = bar_df.head(11)
bar_df = bar_df.drop([5144])

color = ['cornflowerblue', 'skyblue', 'powderblue', 'lightsteelblue', 'blanchedalmond', 'peachpuff', 'lightpink', 'plum', 'sandybrown', 'coral']

states = list(bar_df['state'])[::-1]
total = list(bar_df['tot_cases'])[::-1]
death = list(bar_df['tot_death'])[::-1]

fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(15, 10))
axes[0].barh(states, total, align='center', color=color)
axes[0].set_title('Total Cases Confirmed', fontsize = 14)
axes[1].barh(states, death, align='center', color=color)                                                
axes[1].set_title('Total Deaths', fontsize = 14)

axes[0].invert_xaxis()
axes[0].set(yticks=states, yticklabels=states)
axes[0].yaxis.tick_right()

for ax in axes.flat:
    ax.margins(0.1)
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=13)
    
fig.tight_layout()
ca_patch = mpatches.Patch(color='cornflowerblue', label='California')
tx_patch = mpatches.Patch(color='skyblue', label='Texas')
fl_patch = mpatches.Patch(color='powderblue', label='Florida')
il_patch = mpatches.Patch(color='lightsteelblue', label='Illinois')
ga_patch = mpatches.Patch(color='blanchedalmond', label='Georgia')
nc_patch = mpatches.Patch(color='peachpuff', label='North Carolina')
tn_patch = mpatches.Patch(color='lightpink', label='Tennessee')
az_patch = mpatches.Patch(color='plum', label='Arizona')
ny_patch = mpatches.Patch(color='sandybrown', label='New York')
nj_patch = mpatches.Patch(color='coral', label='New Jersey')
plt.legend(handles=[ca_patch, tx_patch, fl_patch, il_patch, ga_patch, nc_patch, tn_patch, az_patch, ny_patch, nj_patch],
          bbox_to_anchor=(0.5, 0.6), loc='upper left', prop={'size': 16})
fig.suptitle("Confirmed Cases and Death During Covid-19 for Top 10 States", fontsize=20)
fig.subplots_adjust(top=0.85)
plt.show()
fig.savefig("figs/fig7.png")


# 

# The testing results of Covid-19 is also worth noticing. Data shows that there is no testing results reported prior to March. In fact, the first patient who was diagnosed of infection with this virus was reported in January 20th, 2020, and since then the virus starts to spread. From the histogram, there are some reported testing results in March and April, but not too many, which aligns to the fact that the there were only sufficient Covid-19 test kits and teste centers for a limited number of people, and it took a while for the results to be reported. As the pandemic worsens rapidly, more reources such as testing kits, testing centers, medications, medical appliances, and etc are provided to
# people against Covid-19. That's why we see in the plot, the peak occurs in July and August, when more people are concerned about the virus, more resources become available, and the testing results are reported more quickly. In addition, the most number of positive tetsing results are reported in July and August which again confirms with the fact and aligns with the visualization we saw earlier where the reported confirmed cases in the U.S. spiked in July.
# 

# In[54]:


df = pd.read_csv('us_covid19_daily_new.csv')
df = df.rename(columns={'date': 'submission_date'})

fig, axes = plt.subplots(figsize=(20, 10))
month = ['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']


bars1 = list(df['negative'])[::-1]
bars2 = list(df['positive'])[::-1]
# Heights of bars1 + bars2
bars = np.add(bars1, bars2).tolist()
 
# The position of the bars on the x-axis
r = [1,2,3,4, 5, 6, 7]
 
# Names of group and bar width
names = month
barWidth = 1
plt.bar(r, bars1, color='cornflowerblue', edgecolor='white', width=barWidth)
plt.bar(r, bars2, bottom=bars1, color='pink', edgecolor='white', width=barWidth)

plt.xticks(r, names)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Population in Billion', fontsize=14)
plt.title('Covid-19 Testing Results From March to September', fontsize=20)


pos = mpatches.Patch(color='cornflowerblue', label='Positive')
neg = mpatches.Patch(color='pink', label='Negative')
plt.legend(handles=[pos, neg], loc='upper right', bbox_to_anchor=(1, 1), prop={'size': 14})

plt.show()
fig.savefig("figs/fig8.png")


#  

# To digger further, let's break down the confirmed cases and fatality by groups. The first nested donut chart breaks down confirmed cases and fatality by age group. We see that among the four age groups, we see that nearly 68% of people infected with Covid-19 are in the age of 20-60, which makes sense because these people tend to go out more for example, to work and to do grocery, than people of other ages, hence they have a higher chance of getting infection. Only 20% of confirmed cases are from people of age 60+, however they have the largest number of deaths. The second donut chart breaks down deaths by gender, and we see more male died than female. Lastly, we have deaths broken down by race, and we see that 51.3% of patients with Covid-19 are White, followed by Hispanic or Latino. Again, the information we get from these donut charts tie back to the conclusion we made earlier that there are some factors that make some groups of people more easily infected by the virus and less likely to recover from the virus. 

# In[74]:


age = pd.read_csv('age_group.csv')
age = age.drop([4])
gender = pd.read_csv('gender.csv')
race = pd.read_csv('race.csv')

group_names=['12.5%', '37.7%', '29.9%', '20%']
group_size=age['Confirmed']
subgroup_size=[361,408437,2292,1234991,24008,956730,93981,560932]


fig, ax1 = plt.subplots(figsize=(30, 6))
fig.suptitle("Infection and Deaths By Age Group, Gender, and Race", fontsize=20, horizontalalignment='center')
fig.subplots_adjust(top=0.8)

size = 0.3
outer_colors = ['orange', 'yellowgreen', 'navajowhite', 'yellow']
inner_colors = ['tomato', 'forestgreen','tomato', 'forestgreen','tomato', 'forestgreen','tomato', 'forestgreen']

ax1.pie(group_size, radius=1.3, colors=outer_colors, labels=group_names,
       wedgeprops=dict(width=size, edgecolor='w'))

ax1.pie(subgroup_size, radius=1.3-size, colors=inner_colors, #labels=subgroup_names, labeldistance=0.7,
       wedgeprops=dict(width=size, edgecolor='w'), textprops={'fontsize': 16})

ax1.legend(loc='lower right', prop={'size': 16})
ax1.set_title("Infections and Deaths for Different Age Groups", fontsize=16, pad=25, loc='center')

zero_18 = mpatches.Patch(color='orange', label='Age: 0-18')
ninteen_39 = mpatches.Patch(color='yellowgreen', label='Age: 19-39')
forty_59 = mpatches.Patch(color='navajowhite', label='Age: 40-59')
sixty = mpatches.Patch(color='yellow', label='Age: 60+')
recovered = mpatches.Patch(color='forestgreen', label='Recovered')
died = mpatches.Patch(color='tomato', label='Died')

ax1.legend(handles=[zero_18, ninteen_39, forty_59, sixty, recovered, died],
          bbox_to_anchor=(1.3, 1), loc='upper left', prop={'size': 14})


fig, (ax2, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))

group_size = gender['Death']
group_names = ['46.7%', '53.3%']
outer_colors = ['plum', 'powderblue']

ax2.pie(group_size, radius=1.3, colors=outer_colors, labels=group_names,
       wedgeprops=dict(width=size, edgecolor='w'), textprops={'fontsize': 14})

ax2.legend(loc='lower right', prop={'size': 16})
ax2.set_title("Deaths of Gender", fontsize=20, pad=40, loc='center')

female = mpatches.Patch(color='plum', label='Female Death')
male = mpatches.Patch(color='powderblue', label='Male Death')

ax2.legend(handles=[female, male],
          bbox_to_anchor=(1, 1), loc='upper left', prop={'size': 14})


group_size = race['Death']
group_names = ['51.3%', '24.2%', '18.7%', '3.5%', '1.3%', '0.9%']
outer_colors = ['cornflowerblue', 'steelblue', 'lightskyblue', 'navy', 'lightsteelblue', 'lavender']

ax3.pie(group_size, radius=1.3, colors=outer_colors, labels=group_names,
       wedgeprops=dict(width=size, edgecolor='w'), textprops={'fontsize': 14})

ax3.legend(loc='lower right', prop={'size': 16})
ax3.set_title("Deaths of Ethnicity", fontsize=20, pad=40, loc='center')

w = mpatches.Patch(color='cornflowerblue', label='White')
hl = mpatches.Patch(color='steelblue', label='Hispanic or Latino')
b = mpatches.Patch(color='lightskyblue', label='Black')
a = mpatches.Patch(color='navy', label='Asian')
ai = mpatches.Patch(color='lightsteelblue', label='American Indian or Alaska Native')
o = mpatches.Patch(color='lavender', label='Other')

ax3.legend(handles=[w, hl, b, a, ai, o],
          bbox_to_anchor=(1.2, 1), loc='upper left', prop={'size': 14})
fig.tight_layout() 

plt.show()
fig.savefig("figs/fig9.png")


# 

# In[ ]:





# We are also interested in the travels during Covid-19. Let's see the top states people from California and Texas traveled to by observing the network plot. The blue lines represent the travels from Texas, and the pink lines represent the travels from California. The thicker a line, the more travels from the origin states. We see that most travels starting from Texas head to California, Florida, Corolado, and Arizona, since these three blue lines are the thickiest. From California, most travels are to Nevada, Arizona, and Washington. Among these destination states, California, Florida, and Arizona are also in the top 10 states with the most confirmed cases of Covid-19. A resonable assumption drawn from this visualization could be that some people were infected by the virus during their travels, and then they are tested positive at the destination state. 

# In[51]:


df = pd.read_csv('flights_2020.csv')[['ORIGIN_STATE_ABR', 'DEST_STATE_ABR']]

ca = df[df['ORIGIN_STATE_ABR'] == 'CA']
ca = ca.groupby('DEST_STATE_ABR').count()
ca = ca.reset_index()
source = ['CA' for i in range(len(ca))]
target = ca['DEST_STATE_ABR']
count = ca['ORIGIN_STATE_ABR']
data = {'source': source, 'target': target, 'count': count}
ca_df = pd.DataFrame(data)
ca_df = ca_df.sort_values('count', ascending=False)
ca_df = ca_df.head(10)

tx = df[df['ORIGIN_STATE_ABR'] == 'TX']
tx = tx.groupby('DEST_STATE_ABR').count()
tx = tx.reset_index()
source = ['TX' for i in range(len(tx))]
target = tx['DEST_STATE_ABR']
count = tx['ORIGIN_STATE_ABR']
data = {'source': source, 'target': target, 'count': count}
tx_df = pd.DataFrame(data)
tx_df = tx_df.sort_values('count', ascending=False)
tx_df = tx_df.head(10)

network_df = pd.concat([ca_df, tx_df], ignore_index=True)

def add_color(row):
    if row['source'] == 'CA':
        return 'orange'
    if row['source'] == 'TX':
        return 'navy'
def set_edge(row):
    return row['count']/10000

network_df['color'] = network_df.apply(add_color, axis=1)
network_df['width'] = network_df.apply(set_edge, axis=1)

G = nx.Graph()
for index, row in network_df.iterrows():
    G.add_edge(row['source'], row['target'], color=row['color'], weight=row['width'])
    
colors = nx.get_edge_attributes(G,'color').values()
weights = nx.get_edge_attributes(G,'weight').values()

pos = nx.circular_layout(G)
plt.figure(figsize=(15,10))
red_patch = mpatches.Patch(color='orange', label='Flights from California')
blue_patch = mpatches.Patch(color='navy', label='Flights from Texas')
plt.legend(handles=[red_patch, blue_patch], bbox_to_anchor=(0.8, 1), loc='upper left', prop={'size': 16})
plt.title('Top Air Travels Between States During Covid-19 from California and Texas', fontsize=20)
nx.draw(G, pos, 
        edge_color=colors, 
        width=list(weights),
        with_labels=True,
        node_color='lightgray',
       node_size = 1600,
       font_size=12)
plt.savefig("figs/fig10.png")


# In[ ]:





# As many states enforced the stay at home policy, people become more active in social media platforms, for example, Twitter. Let's examine some tweets posted during Covid-19, some very noticeable words that caught our eyes easily are covid19, sanitizers, stop, smelleded, hand, scent, coronavirus, realdonaldtrump, and etc. It seems like many tweets are around the topics of Coronavirus and sanitization. As tremendous people are affected by Coronavirus, people do talk a lot about it on Twitter.

# In[57]:


import nltk 
from wordcloud import WordCloud
import spacy
import re
from nltk.tokenize import TweetTokenizer

df = pd.read_csv("covid19_tweets.csv")

tknzr = TweetTokenizer()
stopwords  = set(nltk.corpus.stopwords.words('english'))
nlp = spacy.load('en')
MAX_CHARS = 20000

def tokenizer(comment):
    comment = comment.lower()
    comment = ''.join(i for i in comment if ord(i)<128)
    comment = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’;#]", " ", str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    if (len(comment) > MAX_CHARS):
        comment = comment[:MAX_CHARS]

    return [x.text for x in nlp.tokenizer(comment) if x.text != " " and x.text not in stopwords and x.text != "n't"]

df['tweet'] = df['text'].apply(tokenizer)

wordcloud = WordCloud(background_color='white',max_words=150,max_font_size=40,scale=3,random_state=50).generate(str(df["tweet"]))

fig = plt.figure(1, figsize=(20, 10))
plt.axis('off')

plt.imshow(wordcloud)
plt.title("Tweets During Covid-19", fontsize=40)
plt.show()
fig.savefig("figs/fig11.png")


# In conclusion, Coronavirus is a tremendous disaster in the human history, and almost everyone in the world is affected in a way. We learned more about the virus from some visualizations created based on the datasets in various aspects. Let's pray that this pandemic will be over soon and no one will suffer from it anymore. 

# In[ ]:





#  

# In[ ]:





# In[ ]:





# In[60]:


df = pd.read_csv('United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv')

def split_month(row):
    month = int(row['submission_date'].split('/')[0])
    return month
def split_date(row):
    date = int(row['submission_date'].split('/')[1])
    return date
df['month'] = df.apply(split_month, axis=1)
df['date'] = df.apply(split_date, axis=1)
df = df[['submission_date', 'month', 'date','state', 'tot_cases', 'tot_death']]
tmp = df[(df['date']==10) | (df['date']==22) | (df['date']==30)]
tmp = tmp[(tmp['state']=='CA') | (tmp['state']=='TX') | (tmp['state']=='FL') | (tmp['state']=='IL') | (tmp['state']=='GA')]


pop = pd.read_csv('population.csv')
tmp2 = pd.merge(map_df, pop, how ='left', left_on=['State'], right_on=['Location'])
pop = tmp2[['Total Cases', 'Children 0-18', 'Adults 19-25', 'Adults 26-34', 'Adults 35-54', 'Adults 55-64', '65+']]
pop['Children'] = pop['Children 0-18']
pop['Adult'] = pop['Adults 19-25'] + pop['Adults 26-34'] + pop['Adults 35-54']
pop['Elderly'] = pop['Adults 55-64'] + pop['65+']

tot_list = list(pop['Total Cases']) * 3
children_list = list(pop['Children'])
adult_list = list(pop['Adult'])
elderly_list = list(pop['Elderly'])
pop_list = children_list + adult_list + elderly_list

category = []
for i in range(51):
    category.append('Children')
for i in range(51):
    category.append('Adult')
for i in range(51):
    category.append('Elderly')
    
data = {'cases': tot_list, 'population': pop_list, 'category': category} 
pop_vs_cases = pd.DataFrame(data) 

fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(25, 10))

sns.scatterplot(x="tot_cases", y="tot_death", data=tmp, hue='state', legend='brief', style='state',
          palette=dict(CA='#BE83CE', TX='#82D3EB', FL='#ACC573', IL='#E59B12', GA='pink'),
          s=100, ax=ax1)
ax1.set_xlabel('Confirmed Cases', fontsize=14)
ax1.set_ylabel('Deaths', fontsize=14)
ax1.legend(loc='lower right', prop={'size': 16})
ax1.set_title("Confirmed Cases Vs Death", fontsize=16)

sns.scatterplot(x="population", y="cases", data=pop_vs_cases, hue='category', legend='brief', style='category',
                palette=dict(Children='#BE83CE', Adult='#82D3EB', Elderly='#ACC573'),
                s=100, ax=ax2)
ax2.set_xlabel('Population in 10 Millions', fontsize=14)
ax2.set_ylabel('Confimed Cases', fontsize=14)
ax2.set_title('Relationship between Age and Confirmed Cases', fontsize=16)

plt.legend(loc='lower right', prop={'size': 14})
fig.suptitle("Relationships Between Confirmed Cases, Deaths, and Age", fontsize=20)
fig.subplots_adjust(top=0.85)
fig.savefig("figs/fig12.png")


# In[ ]:





# In[ ]:




