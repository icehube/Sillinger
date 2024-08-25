import time
import pandas as pd 
import json
import os

from pyscipopt import Model

"""
Todo:
#System is not factoring in players that are starting but are below baseline
#System needs to have a way to factor in salary implications of group ABC
#Make Optimizer use saved version so that I can edit the CSV
#I think optimizer has bids + Salary (which has bids)
"""

# Load teams from the JSON file
with open('teams.json', 'r') as file:
    teams_data = json.load(file)

# Extract penalties from the teams data
PENALTIES = {team: data['penalty'] for team, data in teams_data.items()}

# Constants
SALARY = 56.8
MIN_SALARY = 0.5
TEAMS = 11
FORWARD = 14
DEFENCE = 7
GOALIE = 3

class FantasyAuction:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.players_df = self.load_data()

    def load_data(self):
        try:
            # Load the data into a DataFrame
            df = pd.read_csv(
                self.csv_path, 
                dtype={'PTS': int, 'SALARY': float, 'BID': float}
            )

        except Exception as e:
            print(f"Error reading the CSV file: {e}")
            exit(1)  # Exit the script

        return df

    def process_data(self):
        if self.players_df is None:
            print("Error: No data loaded.")
            return
    
        total_pool = SALARY * TEAMS
        committed_salary = self.players_df[self.players_df['STATUS'] == 'START']['SALARY'].sum()
        total_penalties = sum(PENALTIES.values())   # Calculate the sum of the penalties
        committed_salary += total_penalties     # Add the sum of the penalties to committed_salary
        available_to_spend = total_pool - committed_salary
        self.players_df['Draftable'] = "NO"  # Initialize the Draftable column to 0
        player_count, total_z = self.calculate_z_scores()
        total_bid_sum, restrict, dollar_per_z = self.update_bids(player_count, total_z, available_to_spend)
        return total_pool, committed_salary, available_to_spend, player_count, total_z, total_bid_sum, restrict, dollar_per_z

    def calculate_z_scores(self):

        grouped_players = self.players_df.groupby('POS')

        F_baseline = FORWARD * TEAMS
        D_baseline = DEFENCE * TEAMS
        G_baseline = GOALIE * TEAMS

        player_count = 0  # Initialize the counter
        total_z = 0 

        # Loop over each player group to perform calculations
        for pos, group in grouped_players:
            # Sort the group by points in descending order, and filter out 'MINOR' players
            group = group[group['STATUS'] != 'MINOR'].sort_values('PTS', ascending=False)

            group = group.sort_values('PTS', ascending=False)
            if pos == 'F':
                top_players = group.head(F_baseline)
            elif pos == 'D':
                top_players = group.head(D_baseline)
            elif pos == 'G':
                top_players = group.head(G_baseline)
            else:
                continue

            filtered_top_players = top_players[top_players['FCHL TEAM'].isin(['ENT', 'RFA', 'UFA'])]

            # Use .loc to update the Draftable column in self.players_df for filtered_top_players
            draftable_indices = filtered_top_players.index
            self.players_df.loc[draftable_indices, 'Draftable'] = "YES"

            # Update the counter with the number of rows in filtered_top_players
            player_count += len(filtered_top_players)

            # Calculate Z-scores for filtered top players
            points = filtered_top_players['PTS']
            mean_pts = points.mean()
            stdev_pts = points.std()

            if stdev_pts == 0:
                stdev_pts = 1  # Avoid division by zero

            # Calculate and assign Z-scores directly in the DataFrame
            z_scores = (points - mean_pts) / stdev_pts
            z_scores -= z_scores.min()  # Standardize Z-scores so the minimum is 0

            self.players_df.loc[draftable_indices, 'Z-score'] = z_scores.round(2)

            total_z += z_scores.sum()

        return player_count, total_z
    
    def build_model(self):
        self.model = Model("PlayerSelection")
        self.player_vars = {}

        # Filter players based on specific criteria
        self.filtered_df = self.players_df[self.players_df['FCHL TEAM'].isin(['ENT', 'UFA', 'RFA', 'BOT'])]

        for i, row in self.filtered_df.iterrows():
            player_name = row['PLAYER']
            player_position = row['POS']
            self.player_vars[i] = self.model.addVar(vtype="B", name=f"{player_name}_{player_position}")

        self.model.setObjective(
            sum(row['PTS'] * self.player_vars[i] for i, row in self.filtered_df.iterrows()),
            "maximize"
        )

        self.add_constraints(self.player_vars)

    def solve_model(self):
        self.model.optimize()

    def add_constraints(self, player_vars):
        # Identify players that must be included in the team
        must_include_players = self.filtered_df.index[
            (self.filtered_df['FCHL TEAM'] == 'BOT') & (self.filtered_df['STATUS'] == 'START')
        ]

        # Constraint 1: Sum of the "Bid" values must be under 56.8
        self.model.addCons(
            sum((row['SALARY'] + row['BID']) * player_vars[i] for i, row in self.filtered_df.iterrows()) <= SALARY
        )

        # Constraint 3: Must have X Players with F in their Pos Column
        self.model.addCons(
            sum(player_vars[i] for i, row in self.filtered_df.iterrows() if row['POS'] == 'F') == FORWARD
        )

        # Constraint 4: Must have X Players with D in their Pos Column
        self.model.addCons(
            sum(player_vars[i] for i, row in self.filtered_df.iterrows() if row['POS'] == 'D') == DEFENCE
        )

        # Constraint 5: Must have X Players with G in their Pos Column
        self.model.addCons(
            sum(player_vars[i] for i, row in self.filtered_df.iterrows() if row['POS'] == 'G') == GOALIE
        )

        # Constraint: Must include players from BOT team with status "start"
        for i in must_include_players:
            self.model.addCons(player_vars[i] == 1)
        pass

    def get_solution(self):
        best_solution = self.model.getBestSol()
        return best_solution

    def update_bids(self, player_count, total_z, available_to_spend):
        restrict = player_count * MIN_SALARY
        print(f"Restricted amount: {restrict}")

        dollar_per_z = (available_to_spend - restrict) / total_z
        print(f"Dollar per Z-score: {dollar_per_z}")

        self.players_df.loc[self.players_df['Draftable'] == 'YES', 'BID'] = (self.players_df['Z-score'] * dollar_per_z) + MIN_SALARY
        self.players_df['BID'] = self.players_df['BID'].round(1)

        # Split the DataFrame into three tables based on position
        goalies_df = self.players_df[self.players_df['POS'] == 'G']
        defenders_df = self.players_df[self.players_df['POS'] == 'D']
        forwards_df = self.players_df[self.players_df['POS'] == 'F']

        #Function to print DataFrame with numbering and summary
        def print_table_with_summary(df, position):
             df = df.reset_index(drop=True)
             df.index += 1
             print(f"{position}:")
             print(df)
             print(f"Number of {position} with Bid > 0: {len(df[df['BID'] > 0])}")
             print(f"Number of {position} with Status == 'START': {len(df[df['STATUS'] == 'START'])}")
             print(f"Sum of Bid column for {position}: {df['Bid'].sum()}")
             print("\n")

        # Print the tables with summaries
        # print_table_with_summary(goalies_df, "Goalies")
        # print_table_with_summary(defenders_df, "Defenders")
        # print_table_with_summary(forwards_df, "Forwards")

        total_bid_sum = self.players_df['BID'].sum()
        print(f"Total bid sum: {total_bid_sum}")

        return total_bid_sum, restrict, dollar_per_z

    def write_to_csv(self):
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")  # Current time in seconds since the epoch (1970-01-01 00:00:00)

        # Write updated 'players' list back to a new CSV file
        file_path = f'd:/Dropbox/FCHL/Sillinger/app/players-{current_time}.csv'

        # Write the DataFrame to a new CSV file
        self.players_df.to_csv(file_path, index=False)

    def print_results(self, total_pool, committed_salary, available_to_spend, player_count, total_z, total_bid_sum, restrict, dollar_per_z):

        #print("Optimal value:", self.model.getObjVal())

        print(f"TOTAL_POOL: {total_pool}")

        print(f"COMMITTED_SALARY: {committed_salary:.1f}")

        print(f"AVAILABLE_TO_SPEND: {available_to_spend}")

        print(f"TOTAL_Z: {total_z:.2f}")

        print(f"Players to be auctioned: {player_count}")

        print(f"$ Restriced during the Auciton: {restrict}")

        print(f"DOLLAR_PER_Z: {dollar_per_z:.2f}")

        print('=' * 70)
        print(self.players_df.head(100))
        print("=" * 70)

        print(f"Sum of all Bid Values: {total_bid_sum:.2f}")
        print()
        print("." * 70)
        print()


        # Load teams from teams.json
        with open('teams.json', 'r') as file:
            teams = json.load(file)
        
        # Group players by team
        grouped = self.players_df.groupby('FCHL TEAM')
        
        # Iterate over each team and calculate the summary
        for team_name in teams:
            team_players = grouped.get_group(team_name) 
            
            # Calculate the number of each position where status = 'START'
            start_counts = team_players[team_players['STATUS'] == 'START']['POS'].value_counts()
            num_start_f = start_counts.get('F', 0)
            num_start_d = start_counts.get('D', 0)
            num_start_g = start_counts.get('G', 0)
            
            # Calculate the number of each position where status = 'MINOR'
            minor_counts = team_players[team_players['STATUS'] == 'MINOR']['POS'].value_counts()
            num_minor_f = minor_counts.get('F', 0)
            num_minor_d = minor_counts.get('D', 0)
            num_minor_g = minor_counts.get('G', 0)
            
            # Calculate the sum of salaries
            total_salary = team_players['SALARY'].sum()
            
            # Print the summary for the team
            print(f"{team_name}")
            print(f"  START - F: {num_start_f}, D: {num_start_d}, G: {num_start_g}")
            print(f"  MINOR - F: {num_minor_f}, D: {num_minor_d}, G: {num_minor_g}")
            print(f"  Total Salary: {round(total_salary, 2)}")
            print()
    

if __name__ == "__main__":
    fantasy_auction = FantasyAuction('d:\Dropbox\FCHL\sillinger\data\players-24.csv')
    total_pool, committed_salary, available_to_spend, player_count, total_z, total_bid_sum, restrict, dollar_per_z = fantasy_auction.process_data()
    fantasy_auction.build_model()
    fantasy_auction.print_results(total_pool, committed_salary, available_to_spend, player_count, total_z, total_bid_sum, restrict, dollar_per_z)