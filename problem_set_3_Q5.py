from scipy import stats
import pandas as pd

# Read the data
train = pd.read_csv("/cluster/home/yhtseng/cpl_version_control/train.csv/train.csv")
test = pd.read_csv("/cluster/home/yhtseng/cpl_version_control/test.csv/test.csv")

# Concatenate two pandas objects along a new particular axis
data = pd.concat([train, test], ignore_index = True)

hiv_data = data[data["Disease"] == "HIV"]
dengue_data = data[data["Disease"] == "Dengue"]

hiv_year = hiv_data.groupby("Year")["Count"].sum()
dengue_year = dengue_data.groupby("Year")["Count"].sum()

# We need to align the data to make these two groups have equal length.
aligned_data = pd.DataFrame({"HIV": hiv_year, "Dengue": dengue_year}).fillna(0)

# Spearman's rank correlation
correlation, p_value = stats.spearmanr(aligned_data["HIV"], aligned_data["Dengue"])

print("Correlation:{}\n" "P-value:{}".format(correlation, p_value))
