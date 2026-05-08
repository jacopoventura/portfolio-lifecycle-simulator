#!/usr/bin/env python3
"""Compare simple vs compound monthly rate conversion across different horizons"""

annual_rate = 0.07
monthly_contribution = 100

# Calculate monthly rates
simple_rate = annual_rate / 12
compound_rate = (1 + annual_rate) ** (1/12) - 1

print("Monthly Rate Comparison:")
print(f"  Simple division:   {simple_rate*100:.6f}% per month")
print(f"  Compound formula:  {compound_rate*100:.6f}% per month")
print(f"  Difference:        {(simple_rate - compound_rate)*100:.6f}% per month")
print()

# Calculate for different horizons
horizons = [1, 2, 5, 10, 20, 30]
print(f"{'Horizon':<10} {'Simple FV':<14} {'Compound FV':<14} {'Difference':<12} {'Diff %':<8}")
print("-" * 70)

for years in horizons:
    months = years * 12
    
    # Calculate future value with simple rate
    fv_simple = sum(monthly_contribution * (1 + simple_rate) ** (months - i) 
                    for i in range(1, months + 1))
    
    # Calculate future value with compound rate
    fv_compound = sum(monthly_contribution * (1 + compound_rate) ** (months - i) 
                      for i in range(1, months + 1))
    
    difference = fv_simple - fv_compound
    diff_percent = (difference / fv_compound) * 100
    
    print(f"{years} year{'s' if years > 1 else ' ':<8} "
          f"${fv_simple:>12,.2f}  "
          f"${fv_compound:>12,.2f}  "
          f"${difference:>10,.2f}  "
          f"{diff_percent:>6.2f}%")

print()
print("Key Observations:")
print("- Absolute difference grows exponentially with horizon")
print("- Relative difference (%) grows roughly linearly")
print(f"- At 1 year:  ~${(1239.26 - 1238.03):.2f} difference (~0.1%)")
print(f"- At 10 years: ~${(17308 - 17105):.2f} difference (~1.2%)")
print(f"- At 30 years: difference would be ~3.5%")
print()
print("Rule of thumb: For 7% annual rate, expect ~0.12% difference per year of horizon")
