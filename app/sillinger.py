import time
import pandas as pd 
import json

from pyscipopt import Model

"""
Todo:
#ADD PENALITIES (Remove salary from overall pool)
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
            df = pd.read_csv(
                self.csv_path, 
                dtype={'Pts': int, 'Salary': float, 'Bid': float}
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
        committed_salary = self.players_df[self.players_df['Status'] == 'START']['Salary'].sum()
        total_penalties = sum(PENALTIES.values())   # Calculate the sum of the penalties
        committed_salary += total_penalties     # Add the sum of the penalties to committed_salary
        available_to_spend = total_pool - committed_salary
        player_count, total_z = self.calculate_z_scores()
        total_bid_sum, restrict, dollar_per_z = self.update_bids(player_count, total_z, available_to_spend)
        return total_pool, committed_salary, available_to_spend, player_count, total_z, total_bid_sum, restrict, dollar_per_z

    def calculate_z_scores(self):
        grouped_players = self.players_df.groupby('Pos')

        F_baseline = FORWARD * TEAMS
        D_baseline = DEFENCE * TEAMS
        G_baseline = GOALIE * TEAMS

        player_count = 0  # Initialize the counter
        total_z = 0 

        # Loop over each player group to perform calculations
        for pos, group in grouped_players:
            group = group.sort_values('Pts', ascending=False)
            if pos == 'F':
                top_players = group.head(F_baseline)
            elif pos == 'D':
                top_players = group.head(D_baseline)
            elif pos == 'G':
                top_players = group.head(G_baseline)
            else:
                continue

            filtered_top_players = top_players[top_players['Team'].isin(['ENT', 'RFA', 'UFA'])]

            # Update the counter with the number of rows in filtered_top_players
            player_count += len(filtered_top_players)

            # Calculate Z-scores for filtered top players
            points = filtered_top_players['Pts']
            mean_pts = points.mean()
            stdev_pts = points.std()

            if stdev_pts == 0:
                stdev_pts = 1  # Avoid division by zero

            # Calculate and assign Z-scores directly in the DataFrame
            self.players_df.loc[filtered_top_players.index, 'Z-score'] = (points - mean_pts) / stdev_pts

            # Standardize Z-scores so that the lowest is 0
            min_z_score = self.players_df.loc[filtered_top_players.index, 'Z-score'].min()
            self.players_df.loc[filtered_top_players.index, 'Z-score'] -= min_z_score

            self.players_df['Z-score'].fillna(0, inplace=True)
            self.players_df['Z-score'] = self.players_df['Z-score'].round(2)

            total_z += self.players_df.loc[filtered_top_players.index, 'Z-score'].sum()

            # Update the original DataFrame with the Z-scores calculated
            self.players_df.update(filtered_top_players)

        return player_count, total_z
    
    def build_model(self):
        self.model = Model("PlayerSelection")
        self.player_vars = {}

        # Filter players based on specific criteria
        self.filtered_df = self.players_df[self.players_df['Team'].isin(['ENT', 'UFA', 'RFA', 'BOT'])]

        for i, row in self.filtered_df.iterrows():
            player_name = row['Player']
            player_position = row['Pos']
            self.player_vars[i] = self.model.addVar(vtype="B", name=f"{player_name}_{player_position}")

        self.model.setObjective(
            sum(row['Pts'] * self.player_vars[i] for i, row in self.filtered_df.iterrows()),
            "maximize"
        )

        self.add_constraints(self.player_vars)

    def solve_model(self):
        self.model.optimize()

    def add_constraints(self, player_vars):
        # Identify players that must be included in the team
        must_include_players = self.filtered_df.index[
            (self.filtered_df['Team'] == 'BOT') & (self.filtered_df['Status'] == 'START')
        ]

        # Constraint 1: Sum of the "Bid" values must be under 56.8
        self.model.addCons(
            sum((row['Salary'] + row['Bid']) * player_vars[i] for i, row in self.filtered_df.iterrows()) <= SALARY
        )

        # Constraint 3: Must have X Players with F in their Pos Column
        self.model.addCons(
            sum(player_vars[i] for i, row in self.filtered_df.iterrows() if row['Pos'] == 'F') == FORWARD
        )

        # Constraint 4: Must have X Players with D in their Pos Column
        self.model.addCons(
            sum(player_vars[i] for i, row in self.filtered_df.iterrows() if row['Pos'] == 'D') == DEFENCE
        )

        # Constraint 5: Must have X Players with G in their Pos Column
        self.model.addCons(
            sum(player_vars[i] for i, row in self.filtered_df.iterrows() if row['Pos'] == 'G') == GOALIE
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

        dollar_per_z = (available_to_spend - restrict) / total_z

        self.players_df['Bid'] = (self.players_df['Z-score'] * dollar_per_z) + MIN_SALARY
        self.players_df['Bid'] = self.players_df['Bid'].round(1)

        total_bid_sum = self.players_df[self.players_df['Z-score'] > 0]['Bid'].sum()

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
        print(self.players_df.head(50))
        print("=" * 70)

        print(f"Sum of all Bid Values: {total_bid_sum:.2f}")
    

if __name__ == "__main__":
    fantasy_auction = FantasyAuction('players.csv')
    total_pool, committed_salary, available_to_spend, player_count, total_z, total_bid_sum, restrict, dollar_per_z = fantasy_auction.process_data()
    fantasy_auction.build_model()
    fantasy_auction.print_results(total_pool, committed_salary, available_to_spend, player_count, total_z, total_bid_sum, restrict, dollar_per_z)

    # Test calculate_z_scores method
    player_count, total_z = fantasy_auction.calculate_z_scores()
    print(f"Player Count: {player_count}")
    print(f"Total Z: {total_z}")

    # Print the top 40 players with position 'G' and their Z-scores
    top_40_g = fantasy_auction.players_df[fantasy_auction.players_df['Pos'] == 'F'].sort_values(by='Pts', ascending=False).head(200)
    top_40_g = top_40_g.reset_index(drop=True)
    top_40_g.index = top_40_g.index + 1  # Start the index from 1
    top_40_g.index.name = 'Rank'
    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.expand_frame_repr", True)
    pd.set_option('display.width', 1000)
    print(f"Top 40 G Players and their Z-scores:\n{top_40_g[['Player', 'Pos', 'Team', 'Pts', 'Z-score']]}")
    print(f"Z-scores:\n{fantasy_auction.players_df[['Pos', 'Z-score']].head(50)}")