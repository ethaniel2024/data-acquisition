import pandas as pd

# Load Lahman batting data
batting = pd.read_csv("Batting.csv")

# Filter to Buster Posey
posey = batting[batting["playerID"] == "poseybu01"].copy()

# Keep useful columns
posey = posey[[
    "yearID", "teamID", "lgID", "R", "SO", "CS", "SF", "H", "HBP",
    "HR", "RBI", "2B", "3B", "AB", "G", "BB", "SB"
]].copy()

# Rename columns for readability
posey = posey.rename(columns={
    "yearID": "Season",
    "teamID": "Team",
    "lgID": "League",
    "G": "Games",
    "AB": "At_Bats",
    "R": "Runs",
    "H": "Hits",
    "2B": "Doubles",
    "3B": "Triples",
    "HR": "Home_Runs",
    "RBI": "RBIs",
    "BB": "Walks",
    "SO": "Strikeouts",
    "SB": "Stolen_Bases",
    "CS": "Caught_Stealing",
    "HBP": "Hit_By_Pitch",
    "SF": "Sacrifice_Flies"
})
# Feature engineer traditional measures
posey["Batting_Average"] = posey["Hits"] / posey["At_Bats"]

posey["On_Base_Percentage"] = (
    (posey["Hits"] + posey["Walks"] + posey["Hit_By_Pitch"]) /
    (posey["At_Bats"] + posey["Walks"] + posey["Hit_By_Pitch"] + posey["Sacrifice_Flies"])
)

posey["Singles"] = (
    posey["Hits"] - posey["Doubles"] - posey["Triples"] - posey["Home_Runs"]
)

posey["Total_Bases"] = (
    posey["Singles"] +
    2 * posey["Doubles"] +
    3 * posey["Triples"] +
    4 * posey["Home_Runs"]
)

posey["Slugging_Percentage"] = posey["Total_Bases"] / posey["At_Bats"]
posey["OPS"] = posey["On_Base_Percentage"] + posey["Slugging_Percentage"]

# More feature engineering: per-game metrics
posey["Hits_per_Game"] = posey["Hits"] / posey["Games"]
posey["HR_per_Game"] = posey["Home_Runs"] / posey["Games"]
posey["RBI_per_Game"] = posey["RBIs"] / posey["Games"]
posey["Runs_per_Game"] = posey["Runs"] / posey["Games"]
posey["Walks_per_Game"] = posey["Walks"] / posey["Games"]
posey["Strikeouts_per_Game"] = posey["Strikeouts"] / posey["Games"]
posey["Doubles_per_Game"] = posey["Doubles"] / posey["Games"]
posey["Triples_per_Game"] = posey["Triples"] / posey["Games"]

# Build league-average seasonal comparison dataset
league = batting[[
    "yearID", "R", "H", "HR", "RBI", "2B", "3B", "AB", "G", "BB", "SO"
]].copy()

league = league.rename(columns={
    "yearID": "Season",
    "R": "Runs",
    "H": "Hits",
    "HR": "Home_Runs",
    "RBI": "RBIs",
    "2B": "Doubles",
    "3B": "Triples",
    "AB": "At_Bats",
    "G": "Games",
    "BB": "Walks",
    "SO": "Strikeouts"
})

# Average player-season production each year
league_avg = league.groupby("Season").agg({
    "Runs": "mean",
    "Hits": "mean",
    "Home_Runs": "mean",
    "RBIs": "mean",
    "Doubles": "mean",
    "Triples": "mean",
    "Walks": "mean",
    "Strikeouts": "mean",
    "At_Bats": "mean",
    "Games": "mean"
}).reset_index()

# League average per-game metrics
league_avg["League_Hits_per_Game"] = league_avg["Hits"] / league_avg["Games"]
league_avg["League_HR_per_Game"] = league_avg["Home_Runs"] / league_avg["Games"]
league_avg["League_RBI_per_Game"] = league_avg["RBIs"] / league_avg["Games"]
league_avg["League_Runs_per_Game"] = league_avg["Runs"] / league_avg["Games"]
league_avg["League_Walks_per_Game"] = league_avg["Walks"] / league_avg["Games"]
league_avg["League_Strikeouts_per_Game"] = league_avg["Strikeouts"] / league_avg["Games"]
league_avg["League_Doubles_per_Game"] = league_avg["Doubles"] / league_avg["Games"]
league_avg["League_Triples_per_Game"] = league_avg["Triples"] / league_avg["Games"]

# Keep only the desired league comparison columns
league_avg = league_avg[[
    "Season",
    "League_Hits_per_Game",
    "League_HR_per_Game",
    "League_RBI_per_Game",
    "League_Runs_per_Game",
    "League_Walks_per_Game",
    "League_Strikeouts_per_Game",
    "League_Doubles_per_Game",
    "League_Triples_per_Game"
]]

# Merge league averages into Posey dataset
posey = pd.merge(posey, league_avg, on="Season", how="left")

# Create Posey-vs-League comparison features
posey["Hits_vs_League"] = posey["Hits_per_Game"] - posey["League_Hits_per_Game"]
posey["HR_vs_League"] = posey["HR_per_Game"] - posey["League_HR_per_Game"]
posey["RBI_vs_League"] = posey["RBI_per_Game"] - posey["League_RBI_per_Game"]
posey["Runs_vs_League"] = posey["Runs_per_Game"] - posey["League_Runs_per_Game"]
posey["Walks_vs_League"] = posey["Walks_per_Game"] - posey["League_Walks_per_Game"]
posey["Strikeouts_vs_League"] = posey["Strikeouts_per_Game"] - posey["League_Strikeouts_per_Game"]
posey["Doubles_vs_League"] = posey["Doubles_per_Game"] - posey["League_Doubles_per_Game"]
posey["Triples_vs_League"] = posey["Triples_per_Game"] - posey["League_Triples_per_Game"]

# Three Time WS Champs in THE DYNASTY
posey["Championship_Season"] = posey["Season"].isin([2010, 2012, 2014])

# Sort by season
posey = posey.sort_values("Season").reset_index(drop=True)

# Load postseason batting data
post = pd.read_csv("BattingPost.csv")

# Filter to Posey postseason stats
posey_post = post[post["playerID"] == "poseybu01"].copy()

# Aggregate by season (postseason can have multiple series)
posey_post = posey_post.groupby("yearID").agg({
    "H": "sum",
    "HR": "sum",
    "RBI": "sum",
    "2B": "sum",
    "3B": "sum",
    "AB": "sum",
    "G": "sum",
    "BB": "sum",
    "SO": "sum"
}).reset_index()

# Rename columns
posey_post = posey_post.rename(columns={
    "yearID": "Season",
    "H": "Post_Hits",
    "HR": "Post_HR",
    "RBI": "Post_RBI",
    "2B": "Post_Doubles",
    "3B": "Post_Triples",
    "AB": "Post_AB",
    "G": "Post_Games",
    "BB": "Post_Walks",
    "SO": "Post_Strikeouts"
})

# Create postseason per-game stats
posey_post["Post_Hits_per_Game"] = posey_post["Post_Hits"] / posey_post["Post_Games"]
posey_post["Post_HR_per_Game"] = posey_post["Post_HR"] / posey_post["Post_Games"]
posey_post["Post_RBI_per_Game"] = posey_post["Post_RBI"] / posey_post["Post_Games"]
posey_post["Post_Walks_per_Game"] = posey_post["Post_Walks"] / posey_post["Post_Games"]

# Merge postseason with regular season dataset
posey = pd.merge(
    posey,
    posey_post,
    on="Season",
    how="left"
)

# Compare regular season vs postseason
posey["Hits_Post_vs_Reg"] = posey["Post_Hits_per_Game"] - posey["Hits_per_Game"]
posey["HR_Post_vs_Reg"] = posey["Post_HR_per_Game"] - posey["HR_per_Game"]
posey["RBI_Post_vs_Reg"] = posey["Post_RBI_per_Game"] - posey["RBI_per_Game"]

league_post = post.groupby("yearID").agg({
    "H": "mean",
    "HR": "mean",
    "RBI": "mean",
    "G": "mean"
}).reset_index()

# Create long-form monster dataset
posey_long = posey.melt(
    id_vars=[
        "Season", "Team", "League", "Games", "Championship_Season"
    ],
    value_vars=[
        # Regular season counting stats
        "Hits",
        "Home_Runs",
        "RBIs",
        "Runs",
        "Doubles",
        "Triples",
        "Walks",
        "Strikeouts",
        "Singles",
        "Total_Bases",
        "Stolen_Bases",
        "Caught_Stealing",
        "At_Bats",

        # Regular season rate stats
        "Batting_Average",
        "On_Base_Percentage",
        "Slugging_Percentage",
        "OPS",

        # Regular season per-game stats
        "Hits_per_Game",
        "HR_per_Game",
        "RBI_per_Game",
        "Runs_per_Game",
        "Walks_per_Game",
        "Strikeouts_per_Game",
        "Doubles_per_Game",
        "Triples_per_Game",

        # League averages
        "League_Hits_per_Game",
        "League_HR_per_Game",
        "League_RBI_per_Game",
        "League_Runs_per_Game",
        "League_Walks_per_Game",
        "League_Strikeouts_per_Game",
        "League_Doubles_per_Game",
        "League_Triples_per_Game",

        # Posey relative to league
        "Hits_vs_League",
        "HR_vs_League",
        "RBI_vs_League",
        "Runs_vs_League",
        "Walks_vs_League",
        "Strikeouts_vs_League",
        "Doubles_vs_League",
        "Triples_vs_League",

        # Postseason raw stats
        "Post_Hits",
        "Post_HR",
        "Post_RBI",
        "Post_Doubles",
        "Post_Triples",
        "Post_AB",
        "Post_Games",
        "Post_Walks",
        "Post_Strikeouts",

        # Postseason per-game stats
        "Post_Hits_per_Game",
        "Post_HR_per_Game",
        "Post_RBI_per_Game",
        "Post_Walks_per_Game",

        # Postseason vs regular season
        "Hits_Post_vs_Reg",
        "HR_Post_vs_Reg",
        "RBI_Post_vs_Reg"
    ],
    var_name="Stat",
    value_name="Value"
)

# Add stat family labels
def classify_stat(stat):
    if stat in [
        "Hits", "Home_Runs", "RBIs", "Runs", "Doubles", "Triples",
        "Walks", "Strikeouts", "Singles", "Total_Bases",
        "Stolen_Bases", "Caught_Stealing", "At_Bats"
    ]:
        return "Regular Season Counting Stat"

    elif stat in [
        "Batting_Average", "On_Base_Percentage", "Slugging_Percentage", "OPS"
    ]:
        return "Regular Season Rate Stat"

    elif stat in [
        "Hits_per_Game", "HR_per_Game", "RBI_per_Game", "Runs_per_Game",
        "Walks_per_Game", "Strikeouts_per_Game", "Doubles_per_Game", "Triples_per_Game"
    ]:
        return "Regular Season Per-Game"

    elif stat in [
        "League_Hits_per_Game", "League_HR_per_Game", "League_RBI_per_Game",
        "League_Runs_per_Game", "League_Walks_per_Game",
        "League_Strikeouts_per_Game", "League_Doubles_per_Game",
        "League_Triples_per_Game"
    ]:
        return "League Average Per-Game"

    elif stat in [
        "Hits_vs_League", "HR_vs_League", "RBI_vs_League", "Runs_vs_League",
        "Walks_vs_League", "Strikeouts_vs_League",
        "Doubles_vs_League", "Triples_vs_League"
    ]:
        return "Posey Relative to League"

    elif stat in [
        "Post_Hits", "Post_HR", "Post_RBI", "Post_Doubles", "Post_Triples",
        "Post_AB", "Post_Games", "Post_Walks", "Post_Strikeouts"
    ]:
        return "Postseason Counting Stat"

    elif stat in [
        "Post_Hits_per_Game", "Post_HR_per_Game",
        "Post_RBI_per_Game", "Post_Walks_per_Game"
    ]:
        return "Postseason Per-Game"

    elif stat in [
        "Hits_Post_vs_Reg", "HR_Post_vs_Reg", "RBI_Post_vs_Reg"
    ]:
        return "Postseason vs Regular Season"

    else:
        return "Other"

posey_long["Stat_Type"] = posey_long["Stat"].apply(classify_stat)

# Export files
posey.to_csv("posey_master_wide.csv", index=False)
posey_long.to_csv("posey_master_long.csv", index=False)

print("Wide master dataset saved as: posey_master_wide.csv")
print("Long master dataset saved as: posey_master_long.csv")

print("\nWide dataset preview:")
print(posey.head())

print("\nLong dataset preview:")
print(posey_long.head())