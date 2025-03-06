from scipy import stats
import pandas as pd

# Create data
data = {
    "log_home_range" : [-1.3, -0.5, -0.3, 0.2, 0.1, 0.5, 1.0, 0.3, 0.4, 0.5, 0.1, 0.2,
                        0.4, 1.3, 1.2, 1.4, 1.6, 1.6, 1.8, 3.1],
    "captive_infant" : [4, 22, 0, 0, 11, 13, 17, 25, 24, 27, 29, 33, 33, 42, 33, 20,
                        19, 25, 25, 65]   
}

df = pd.DataFrame(data)

# IQR method
Q1 = df["captive_infant"].quantile(0.25)
Q3 = df["captive_infant"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify potential outliers
outliers = df[(df["captive_infant"] < lower_bound) | (df["captive_infant"] > upper_bound)]

# Exclude potential outliers
df_without_outliers = df[~outliers]

# Calculate linear regression statistics
slop, pvalue, std_err = stats.linregress(df["log_home_range"], df["captive_infant"])
slop_no_out, pvalue_no_out, std_err_no_out = stats.linregress(df_without_outliers["log_home_range"], df_without_outliers["captive_infant"])

print("With outliers:\n" "slop:{}\n" "p-value:{}\n" "standard err:{}\n".format(slop, pvalue, std_err))
print("Without outliers:\n" "slop:{}\n" "p-value:{}\n" "standard err:{}\n".format(slop_no_out, pvalue_no_out, std_err_no_out))
