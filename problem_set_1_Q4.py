import pandas as pd
from scipy.stats import chi2_contingency

# [[# of men admitted, # of women admitted],[# of men not admitted, # of women not admitted]]
Department = {
    'A': [[825, 108], [933-825, 933-108]],
    'B': [[560, 25],[585-560, 585-25]],
    'C': [[325, 593],[918-325, 918-593]],
    'D': [[417, 375],[792-417, 792-375]],
    'F': [[373, 341],[714-373, 714-341]],
    'E': [[191, 393],[584-191, 584-393]],
    'F': [[373, 341],[714-373, 714-393]]    
}

for depart, data in Department.items():
    chi2, p, dof, ex = chi2_contingency(data)
    print(f"Department {depart}: Chi2 = {chi2}, P-value = {p}, DOF = {dof}")

# Toatal number
total_men       = 2691
total_women     = 1835

men_admitted    = 2691*0.45
women_admitted  = 1835*0.3

men_not_admit   = total_men - men_admitted
women_not_admit = total_women - women_admitted

contingency_table = [[men_admitted, men_not_admit], [women_admitted, women_not_admit]]

chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"Chi2 : {chi2}")
print(f"P-value : {p}")
print(f"DOF : {dof}")
print(f"Expected frequency : {expected}")
