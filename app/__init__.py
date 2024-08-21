import csv
import statistics
import time
import pandas as pd 

from pyscipopt import Model

SALARY = 56.8 
MIN_SALARY = 0.5
TEAMS = 11
FORWARD = 14
DEFENCE = 7
GOALIE = 3
#BENCH = 4
#ADD PENALITIES (Remove salary from overall pool)
#Make Optimizer use saved version so that I can edit the CSV

F_BASELINE = FORWARD * TEAMS
D_BASELINE = DEFENCE * TEAMS
G_BASELINE = GOALIE * TEAMS

TOTAL_POOL = SALARY * TEAMS
DISCRETIONARY = TOTAL_POOL - ((F_BASELINE + D_BASELINE + G_BASELINE) * MIN_SALARY)

current_time = time.strftime("%Y-%m-%d_%H-%M-%S")  # Current time in seconds since the epoch (1970-01-01 00:00:00)
print(f"current_time:{current_time}")
print("="*75)

# Read the CSV file into a DataFrame
players_df = pd.read_csv('d:/Dropbox/FCHL/Sillinger/app/players.csv', header=0)

# Filter out invalid or missing points
players_df['Pts'] = players_df['Pts'].astype(int)
players_df['Salary'] = players_df['Salary'].astype(float)
players_df['Bid'] = players_df['Bid'].astype(float)

# Calculate COMMITTED_SALARY
COMMITTED_SALARY = players_df[players_df['Status'] == 'START']['Salary'].sum()

# Initialize grouped_players DataFrame to hold players by their positions
grouped_players = players_df.groupby('Pos')

print(f"TOTAL_POOL: {TOTAL_POOL}")

print(f"COMMITTED_SALARY: {COMMITTED_SALARY:.1f}")

AVAILABLE_TO_SPEND = TOTAL_POOL - COMMITTED_SALARY

print(f"AVAILABLE_TO_SPEND: {AVAILABLE_TO_SPEND}")

TOTAL_Z = 0.0

# Initialize grouped_players DataFrame to hold players by their positions
grouped_players = players_df.groupby('Pos')

player_count = 0  # Initialize the counter

# Loop over each player group to perform calculations
for pos, group in grouped_players:
    group = group.sort_values('Pts', ascending=False)
    if pos == 'F':
        top_players = group.head(F_BASELINE)
    elif pos == 'D':
        top_players = group.head(D_BASELINE)
    elif pos == 'G':
        top_players = group.head(G_BASELINE)
    else:
        continue

    filtered_top_players = top_players[top_players['Team'].isin(['ENT', 'RFA', 'UFA'])]

    # Update the counter with the number of rows in filtered_top_players
    player_count += len(filtered_top_players)

    # Calculate Z-scores for filtered top players
    points = filtered_top_players['Pts']
    mean_pts = points.mean()
    stdev_pts = points.std()

    # Calculate and assign Z-scores directly in the DataFrame
    players_df.loc[filtered_top_players.index, 'Z-score'] = (points - mean_pts) / stdev_pts

    # Standardize Z-scores so that the lowest is 0
    min_z_score = players_df.loc[filtered_top_players.index, 'Z-score'].min()
    players_df.loc[filtered_top_players.index, 'Z-score'] -= min_z_score

    players_df['Z-score'].fillna(0, inplace=True)
    players_df['Z-score'] = players_df['Z-score'].round(2)

    TOTAL_Z += players_df.loc[filtered_top_players.index, 'Z-score'].sum()

# Update the original DataFrame with the Z-scores calculated
players_df.update(filtered_top_players)

print(f"TOTAL_Z: {TOTAL_Z:.2f}")
print(f"PLayers to be auctioned: {player_count}")

RESTRICT = player_count * MIN_SALARY

print(f"$ Restriced during the Auciton: {RESTRICT}")

DOLLAR_PER_Z = (AVAILABLE_TO_SPEND - RESTRICT) / TOTAL_Z

print(f"DOLLAR_PER_Z: {DOLLAR_PER_Z:.2f}")

# Update "Bid" column
players_df['Bid'] = (players_df['Z-score'] * DOLLAR_PER_Z) + MIN_SALARY
players_df['Bid'] = players_df['Bid'].round(1)

# Underline the header
print('=' * 70)
print(players_df.head(30))
print("=" * 70)

# Print the sum of all Bid 
total_bid_sum = players_df[players_df['Z-score'] > 0]['Bid'].sum()
print(f"Sum of all Bid Values: {total_bid_sum:.2f}")

# Write updated 'players' list back to a new CSV file
file_path = f'd:/Dropbox/FCHL/Sillinger/app/players-{current_time}.csv'

# Write the DataFrame to a new CSV file
players_df.to_csv(file_path, index=False)

# Filter the DataFrame
filtered_df = players_df[players_df['Team'].isin(['ENT', 'UFA', 'RFA', 'BOT'])]


# Initialize model
model = Model("PlayerSelection")

# Create variables
player_vars = {}
for i, row in filtered_df.iterrows():
    player_name = row['Player']
    player_position = row['Pos']
    player_vars[i] = model.addVar(vtype="B", name=f"{player_name}_{player_position}")


# Objective function: Maximize the sum of the "Pts" values
model.setObjective(
    sum(row['Pts'] * player_vars[i] for i, row in filtered_df.iterrows()),
    "maximize",
)

# Identify players that must be included in the team
must_include_players = filtered_df.index[
    (filtered_df['Team'] == 'BOT') & (filtered_df['Status'] == 'START')
]

# Constraint 1: Sum of the "Bid" values must be under 56.8
model.addCons(
    sum((row['Salary'] + row['Bid']) * player_vars[i] for i, row in filtered_df.iterrows()) <= SALARY
)

# Constraint 3: Must have X Players with F in their Pos Column
model.addCons(
    sum(player_vars[i] for i, row in filtered_df.iterrows() if row['Pos'] == 'F') == FORWARD
)

# Constraint 4: Must have X Players with D in their Pos Column
model.addCons(
    sum(player_vars[i] for i, row in filtered_df.iterrows() if row['Pos'] == 'D') == DEFENCE
)

# Constraint 5: Must have X Players with G in their Pos Column
model.addCons(
    sum(player_vars[i] for i, row in filtered_df.iterrows() if row['Pos'] == 'G') == GOALIE
)

# Constraint: Must include players from BOT team with status "start"
for i in must_include_players:
    model.addCons(player_vars[i] == 1)

# Solve model
model.optimize()

# Print results
print("Optimal value:", model.getObjVal())

best_solution = model.getBestSol()

# Initialize an empty list to collect players
all_players = []

# Collect and sort players
for i, row in filtered_df.iterrows():
    if model.getVal(player_vars[i]) > 0.5:
        reduced_cost = model.getVarRedcost(player_vars[i])
        player_data = row.to_dict()
        player_data['Reduced Cost'] = reduced_cost
        all_players.append(player_data)

# Convert it back to a DataFrame if you wish
all_selected_players_df = pd.DataFrame(all_players)

# Sort by position and points
sorted_df = all_selected_players_df.sort_values(by=['Pos', 'Pts'], ascending=[True, False])

# Initialize empty DataFrames for each position
forward_players = pd.DataFrame()
defense_players = pd.DataFrame()
goalie_players = pd.DataFrame()
other_players = pd.DataFrame()

# Populate the DataFrames
forward_players = sorted_df[(sorted_df['Pos'] == 'F')].head(12)
defense_players = sorted_df[(sorted_df['Pos'] == 'D')].head(6)
goalie_players = sorted_df[(sorted_df['Pos'] == 'G')].head(2)

# Take the index of players that are either Forward, Defense, or Goalie and are in the top N spots
taken_indices = pd.concat([forward_players, defense_players, goalie_players]).index

# Take all players that are NOT in the taken spots
other_players = sorted_df[~sorted_df.index.isin(taken_indices)]

# Combine all DataFrames
sorted_players = pd.concat([forward_players, defense_players, goalie_players, other_players], ignore_index=True)

# Initialize variables to hold the total points, total bid, and total salary
total_pts = sorted_players['Pts'].sum()
total_bid = sorted_players['Bid'].sum()
total_salary = sorted_players['Salary'].sum()

# Print sorted players in a formatted table
print(f"{'Pos':<5}{'Player':<25}{'Team':<10}{'Pts':<10}{'Bid':<10}{'Salary':<10}{'Reduced Cost':<15}")

# Line separator for header and content
print("-"*75)

for index, row in sorted_players.iterrows():
    print(f"{row['Pos']:<5}{row['Player']:<25}{row['Team']:<10}{row['Pts']:<10}{row['Bid']:<10}{row['Salary']:<10}{row['Reduced Cost']:.2f}")

# Print the total points and total bid at the end
print("-"*75)  # Line separator for content and total
print(f"{'Total':<40}{total_pts:<10}{total_bid:<10}{total_salary:<10}")