import time
import pandas as pd 
import json
from tabulate import tabulate

from pyscipopt import Model

"""
Todo:
# Fix Optimizer
# Import Dobber Projection/Evolving Hockey
# Colour Bot Players and Prints
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
                dtype={'AGE': int, 'PTS': int, 'SALARY': float, 'BID': float}
            )

        except Exception as e:
            print(f"Error reading the CSV file: {e}")
            exit(1)  # Exit the script

        return df

    def process_data(self):
        if self.players_df is None:
            print("Error: No data loaded.")
            return
    
        # Initialize the Draftable column to NO
        self.players_df['Draftable'] = "NO"  
        # Fill missing values in the 'STATUS' column with 'NO'
        self.players_df['STATUS'] = self.players_df['STATUS'].fillna('NO')
        # Set the salary of players with 'FCHL TEAM' as 'RFA', 'UFA', or 'ENT' to 0
        self.players_df.loc[self.players_df['FCHL TEAM'].isin(['RFA', 'UFA', 'ENT']), 'SALARY'] = 0

        total_pool = SALARY * TEAMS
        # Calculate the sum of the salaries of players with 'STATUS' as 'START' or 'MINOR' and 'GROUP' as 2 or 3
        committed_salary = self.players_df[
            (self.players_df['STATUS'] == 'START') |
            ((self.players_df['STATUS'] == 'MINOR') & (self.players_df['GROUP'].isin(['2', '3'])))
        ]['SALARY'].sum()
        # Calculate the sum of the penalties
        total_penalties = sum(PENALTIES.values())   
        # Add the sum of the penalties to committed_salary
        committed_salary += total_penalties     
        available_to_spend = total_pool - committed_salary
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

            if pos == 'F':
                baseline = F_baseline
            elif pos == 'D':
                baseline = D_baseline
            elif pos == 'G':
                baseline = G_baseline
            else:
                continue

            print(f"Initial {pos} Baseline: {baseline}")

            # Slice the DataFrame first, then apply the boolean condition
            below_baseline_slice = group.iloc[baseline:]
            below_baseline_start_players = below_baseline_slice[below_baseline_slice['STATUS'] == 'START']
            count_below_baseline_start = below_baseline_start_players.shape[0]

            print(f"Count Below Baseline for {pos}: {count_below_baseline_start}")

            # Adjust the baseline
            adjusted_baseline = baseline - count_below_baseline_start
            print(f"Adjusted Baseline for {pos}: {adjusted_baseline}")


            # Select the top players using the adjusted baseline
            top_players = group.head(adjusted_baseline)

            # Further processing with top_players if needed
            #with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False):
                #print(top_players[['PLAYER', 'POS', 'PTS', 'FCHL TEAM']])
                #print("\n")

            # if pos == 'F':
            #     top_players = group.head(F_baseline)
            # elif pos == 'D':
            #     top_players = group.head(D_baseline)
            # elif pos == 'G':
            #     top_players = group.head(G_baseline)
            # else:
            #     continue

            # Filter out players with 'FCHL TEAM' as 'ENT', 'RFA', or 'UFA'
            filtered_top_players = top_players[top_players['FCHL TEAM'].isin(['ENT', 'RFA', 'UFA'])]

            # Use .loc to update the Draftable column in self.players_df for filtered_top_players
            draftable_indices = filtered_top_players.index
            self.players_df.loc[draftable_indices, 'Draftable'] = "YES"

            #print(f"Filtered {pos} Players:")
            #print(filtered_top_players)

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

        # Filter players based on specific criteria and remove players with Bid = 0
        self.filtered_df = self.players_df[
            ((self.players_df['FCHL TEAM'].isin(['ENT', 'UFA', 'RFA'])) & (self.players_df['BID'] > 0)) |
            ((self.players_df['FCHL TEAM'] == 'BOT') & (self.players_df['STATUS'] == 'START'))
        ]

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
        try:
            self.model.optimize()
            status = self.model.getStatus()
            if status == "optimal":
                return self.get_solution()
            elif status == "timelimit":
                print("Warning: Optimization hit the time limit.")
            else:
                print(f"Warning: The model did not solve to optimality. Status: {status}")
            return None
        except ValueError as ve:
            print(f"ValueError during optimization: {ve}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during optimization: {e}")
            return None

    def add_constraints(self, player_vars):
        if self.filtered_df.empty:
            print("Error: No players to consider in the optimization.")
            return

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
        #print(f"Restricted amount: {restrict}")

        dollar_per_z = (available_to_spend - restrict) / total_z
        #print(f"Dollar per Z-score: {dollar_per_z}")

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
             with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False):
                print("=" * 90)
                print("=" * 90)
                print(f"{position}:")
                print(df)
                print(f"Number of {position} with Bid > 0: {len(df[df['BID'] > 0])}")
                print(f"Number of {position} with Status == 'START': {len(df[df['STATUS'] == 'START'])}")
                print(f"Sum of Bid column for {position}: {df['BID'].sum()}")
                print("\n")

        # Print the tables with summaries
        print_table_with_summary(forwards_df, "Forwards")
        print_table_with_summary(defenders_df, "Defenders")
        print_table_with_summary(goalies_df, "Goalies")

        total_bid_sum = self.players_df['BID'].sum()
        #print(f"Total bid sum: {total_bid_sum}")

        return total_bid_sum, restrict, dollar_per_z

    def write_to_csv(self):
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")  # Current time in seconds since the epoch (1970-01-01 00:00:00)

        # Write updated 'players' list back to a new CSV file
        file_path = f'd:/Dropbox/FCHL/Sillinger/app/players-{current_time}.csv'

        # Write the DataFrame to a new CSV file
        self.players_df.to_csv(file_path, index=False)

    def print_results(self, total_pool, committed_salary, available_to_spend, player_count, total_z, total_bid_sum, restrict, dollar_per_z, best_solution):

        print(f"TOTAL POOL: {total_pool}")

        print(f"COMMITTED SALARY: {committed_salary:.1f}")

        print(f"AVAILABLE TO SPEND: {available_to_spend:.1f}")

        print(f"TOTAL Z: {total_z:.2f}")

        print(f"Players to be auctioned: {player_count}")

        print(f"$ Restriced during the Auciton: {restrict}")

        print(f"DOLLAR_PER_Z: {dollar_per_z:.2f}")

        print('=' * 90)

        #with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False):
            #print(self.players_df.drop(columns=['NHL TEAM', 'AGE']).head(100))

        print("=" * 90)
        print(f"Sum of all Bid Values: {total_bid_sum:.1f}")
        print()
        print("." * 90)
        print()


        # Load teams from teams.json
        with open('teams.json', 'r') as file:
            teams = json.load(file)
        
        # Group players by team
        grouped = self.players_df.groupby('FCHL TEAM')

        # Function to format tables side by side
        def format_side_by_side(table1, table2):
            table1_lines = table1.split('\n')
            table2_lines = table2.split('\n')
            
            max_lines = max(len(table1_lines), len(table2_lines))
            table1_lines += [''] * (max_lines - len(table1_lines))
            table2_lines += [''] * (max_lines - len(table2_lines))
            
            combined_lines = [f"{line1:<60} {line2}" for line1, line2 in zip(table1_lines, table2_lines)]
            return '\n'.join(combined_lines)

        # Custom sorting key for positions
        def position_sort_key(pos):
            if pos == 'F':
                return 0
            elif pos == 'D':
                return 1
            elif pos == 'G':
                return 2
            return 3

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

            # Calculate the total number of players with STATUS = 'START' and 'MINOR'
            total_start_players = team_players[team_players['STATUS'] == 'START'].shape[0]
            total_minor_players = team_players[team_players['STATUS'] == 'MINOR'].shape[0]

            # Calculate the sum of points for players with STATUS = 'START'
            total_pts_start = team_players[team_players['STATUS'] == 'START']['PTS'].sum()
            
            # Calculate the sum of salaries
            total_salary = team_players[
                (team_players['STATUS'] == 'START') | 
                ((team_players['STATUS'] == 'MINOR') & (team_players['GROUP'].isin(["2", "3"])))
            ]['SALARY'].sum()

            # Prepare the summary table
            summary_table = [
                    ["START - F", num_start_f],
                    ["START - D", num_start_d],
                    ["START - G", num_start_g],
                    ["MINOR - F", num_minor_f],
                    ["MINOR - D", num_minor_d],
                    ["MINOR - G", num_minor_g],
                    ["-"*18, "-"*4],  # Separator line
                    ["Total START Players", total_start_players],
                    ["Total MINOR Players", total_minor_players],
                    ["Total PTS (START)", total_pts_start],
                    ["-"*18, "-"*4],  # Separator line
                    ["Total Salary", round(total_salary, 2)]
                ]

            # Sort the players by STATUS, custom POS order, and PTS
            sorted_players = team_players.sort_values(
                by=['STATUS', 'POS', 'PTS'], 
                ascending=[False, True, False], 
                key=lambda col: col.map(position_sort_key) if col.name == 'POS' else col
            )
            
            # Prepare the players table
            players_table = sorted_players[['PLAYER', 'GROUP', 'POS', 'STATUS', 'PTS', 'SALARY']].values.tolist()
            
            # Convert tables to string format
            players_table_str = tabulate(players_table, headers=["Player", "Pos", "Status", "Pts", "Salary"], tablefmt="fancy_outline")
            summary_table_str = tabulate(summary_table, headers=["Category", "Count"], tablefmt="fancy_outline")
    
            # Print the summary for the team
            #print("-" * 110)
            #print(f"{team_name}")
            #print("-" * 110)
            #print(format_side_by_side(players_table_str, summary_table_str))
            #print()

        solution_data = []
        for i, row in self.filtered_df.iterrows():
            if best_solution[self.player_vars[i]] > 0.5:  # If the player is selected in the solution
                solution_data.append([
                    row['PLAYER'],
                    row['POS'],
                    row['NHL TEAM'],
                    row['STATUS'],
                    row['PTS'],
                    row['SALARY'],
                    row['BID']
                ])

        headers = ["Player", "Position", "NHL Team", "Status", "Points", "Salary", "Bid"]

        print("Optimized Team for BOT:")
        print(tabulate(solution_data, headers=headers, tablefmt="fancy_outline"))

if __name__ == "__main__":
    # Instantiate the FantasyAuction class
    fantasy_auction = FantasyAuction('d:\Dropbox\FCHL\sillinger\data\players-24.csv')

    # Process the data to set up everything needed for the auction
    total_pool, committed_salary, available_to_spend, player_count, total_z, total_bid_sum, restrict, dollar_per_z = fantasy_auction.process_data()

    # Build and solve the model before printing results
    fantasy_auction.build_model()
    best_solution = fantasy_auction.solve_model()

    # Print the results
    if best_solution is not None:
        fantasy_auction.print_results(total_pool, committed_salary, available_to_spend, player_count, total_z, total_bid_sum, restrict, dollar_per_z, best_solution)
