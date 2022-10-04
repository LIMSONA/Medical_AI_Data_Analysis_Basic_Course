import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

world_cups = pd.read_csv("WorldCups.csv")
print(world_cups[["Year","Attendance"]])

#===============================

world_cups = pd.read_csv("WorldCups.csv")

world_cups = world_cups[['Year', 'Attendance']]
print(world_cups)

plt.plot(world_cups['Year'], world_cups['Attendance'], marker='o', color='black')

#===============================
df=pd.read_csv("WorldCups.csv")
tmp=df[["Year","GoalsScored","MatchesPlayed"]]

tmp["GoalsPerMatch"]= tmp["GoalsScored"]/tmp["MatchesPlayed"]

print(tmp)

#===============================
world_cups = world_cups[['Year', 'GoalsScored', 'MatchesPlayed']]
world_cups["GoalsPerMatch"] = world_cups["GoalsScored"] / world_cups["MatchesPlayed"]


# 첫 번째 그래프 출력
fig, axes = plt.subplots(2, 1, figsize=(4,8))

axes[0].bar(x=world_cups['Year'], height=world_cups['GoalsScored'], color='grey', label='goals')

axes[0].plot(world_cups['Year'], world_cups['MatchesPlayed'], marker='o', color='blue', label='matches')

axes[0].legend(loc='upper left')


# 두 번째 그래프 출력
axes[1].grid(True)
axes[1].plot(world_cups['Year'], world_cups['GoalsPerMatch'], marker='o', color='red', label='goals_per_matches')

axes[1].legend(loc='lower left')

#===============================
df=pd.read_csv("WorldCupMatches.csv")
world_cups_matches = df.replace(
    {'Germany FR': 'Germany',
    "C�te d'Ivoire":"Côte d'Ivoire",
    "rn”>Bosnia and Herzegovina" : "Bosnia and Herzegovina",
    "rn”>Serbia and Montenegro": "Serbia and Montenegro",
    "rn”>United Arab Emirates":"United Arab Emirates",
    "Soviet Union":"Russia"})


world_cups_matches = world_cups_matches.drop_duplicates()

#===============================
world_cups_matches = preprocess.world_cups_matches

#print(world_cups_matches.info())
#홈팀 총득점
home = world_cups_matches.groupby(['Home Team Name'])['Home Team Goals'].sum()

#어웨이팀 총득점
away = world_cups_matches.groupby(['Away Team Name'])['Away Team Goals'].sum()


#goal_per_country= pd.concat([home,away],axis=1).fillna(0)
goal_per_country = pd.concat([home, away], axis=1, sort=True).fillna(0)

goal_per_country["Goals"] = goal_per_country["Home Team Goals"] + goal_per_country["Away Team Goals"]

goal_per_country = goal_per_country["Goals"].sort_values(ascending = False)

goal_per_country = goal_per_country.astype(int)

print(goal_per_country)

#===============================
goal_per_country = preprocess.goal_per_country
goal_per_country = goal_per_country[:10]

# x, y값 저장
x = goal_per_country.index
y = goal_per_country.values

#그래프 그리기
fig, ax = plt.subplots()

ax.bar(x, y, width = 0.5)

# x축 항목 이름 지정, 30도 회전
plt.xticks(x, rotation=30)
plt.tight_layout()

#===============================
world_cups_matches = preprocess.world_cups_matches

world_cups_matches = world_cups_matches[world_cups_matches['Year']==2014]

home_team_goal = world_cups_matches.groupby(['Home Team Name'])['Home Team Goals'].sum()
away_team_goal = world_cups_matches.groupby(['Away Team Name'])['Away Team Goals'].sum()

team_goal_2014 = pd.concat([home_team_goal, away_team_goal], axis=1).fillna(0)


team_goal_2014['goals'] = team_goal_2014['Home Team Goals'] + team_goal_2014['Away Team Goals']
team_goal_2014 = team_goal_2014.drop(['Home Team Goals', 'Away Team Goals'], axis=1)

team_goal_2014.astype('int')
team_goal_2014 = team_goal_2014['goals'].sort_values(ascending=False)

print(team_goal_2014)

#===============================
team_goal_2014.plot(x=team_goal_2014.index, y=team_goal_2014.values, kind="bar", figsize=(12, 12), fontsize=14)

fig, ax = plt.subplots()
ax.bar(team_goal_2014.index, team_goal_2014.values)
plt.xticks(rotation = 90)
plt.tight_layout()

#===============================
world_cups = pd.read_csv("WorldCups.csv")


winner = world_cups["Winner"]
runners_up = world_cups["Runners-Up"]
third = world_cups["Third"]
fourth = world_cups["Fourth"]

winner_count = pd.Series(winner.value_counts())
runners_up_count = pd.Series(runners_up.value_counts())
third_count = pd.Series(third.value_counts())
fourth_count = pd.Series(fourth.value_counts())

ranks = pd.DataFrame({
  "Winner" : winner_count,
  "Runners_Up" : runners_up_count,
  "Third" : third_count,
  "Fourth" : fourth_count
})

ranks = ranks.fillna(0).astype('int64')

ranks = ranks.sort_values(['Winner', 'Runners_Up', 'Third', 'Fourth'], ascending=False)

print(ranks)

#===============================
# x축에 그려질 막대그래프들의 위치입니다.
x = np.array(list(range(0, len(ranks))))

# 그래프를 그립니다.
fig, ax = plt.subplots()

# x 위치에, 항목 이름으로 ranks.index(국가명)을 붙입니다.
plt.xticks(x, ranks.index, rotation=90)
plt.tight_layout()

# 4개의 막대를 차례대로 그립니다.
ax.bar(x - 0.3, ranks['Winner'],     color = 'gold',   width = 0.2, label = 'Winner')
ax.bar(x - 0.1, ranks['Runners_Up'], color = 'silver', width = 0.2, label = 'Runners_Up')
ax.bar(x + 0.1, ranks['Third'],      color = 'brown',  width = 0.2, label = 'Third')
ax.bar(x + 0.3, ranks['Fourth'],     color = 'black',  width = 0.2, label = 'Fourth')
