#!/usr/bin/env python3
"""
Example usage of the BeV (Betriebliche Altersversorgung) calculator
"""

from src.bev import BeV


if __name__ == "__main__":
    print("=" * 70)
    print("BETRIEBLICHE ALTERSVERSORGUNG (BeV) - BOSCH DEUTSCHLAND")
    print("=" * 70)
    
    # Example: Employee who is currently 30 years old, planning until retirement at 67
    # That's 37 years total, with 25 years until age 55
    bev = BeV(
        total_horizon=37,              # 37 years until retirement
        horizon_to_55_year_old=25,     # 25 years until age 55
        initial_balance=5000,          # Starting with €5,000
        yearly_company_contribution=1200,  # Company adds €1,200/year
        monthly_contribution=300       # Employee contributes €300/month
    )
    
    result = bev.calculate_balance()
    
    print(f"\nInitial balance: €{bev.initial_balance:,.2f}")
    print(f"Monthly employee contribution: €{bev.monthly_contribution:,.2f}")
    print(f"Yearly company contribution: €{bev.yearly_company_contribution:,.2f}")
    print(f"Total horizon: {bev.total_horizon} years")
    print(f"Years until age 55: {bev.horizon_to_55_year_old}")
    print(f"\nGrowth rates (net of {bev.ter*100:.2f}% TER):")
    print(f"  - Until age 55: {bev.net_growth_rate_before_55*100:.2f}%")
    print(f"  - After age 55: {bev.net_growth_rate_after_55*100:.2f}%")
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total invested: €{result['total_invested']:,.2f}")
    print(f"  - Employee contributions: €{result['total_employee_contributions']:,.2f}")
    print(f"  - Company contributions: €{result['total_company_contributions']:,.2f}")
    print(f"\nFinal balance: €{result['final_balance']:,.2f}")
    print(f"Total gains: €{result['total_gains']:,.2f}")
    print(f"Return on investment: {(result['total_gains'] / result['total_invested'] * 100):.2f}%")
    
    # Show first 5 and last 5 years
    print("\n" + "=" * 70)
    print("YEARLY DETAILS (First 5 years)")
    print("=" * 70)
    for detail in result['yearly_details'][:5]:
        print(f"Year {detail['year']:2d}: "
              f"€{detail['balance_start']:>10,.2f} → €{detail['balance_end']:>10,.2f} "
              f"(+€{detail['gains']:>8,.2f} gains, {detail['growth_rate']*100:.1f}% rate)")
    
    print("\n" + "=" * 70)
    print("YEARLY DETAILS (Last 5 years)")
    print("=" * 70)
    for detail in result['yearly_details'][-5:]:
        print(f"Year {detail['year']:2d}: "
              f"€{detail['balance_start']:>10,.2f} → €{detail['balance_end']:>10,.2f} "
              f"(+€{detail['gains']:>8,.2f} gains, {detail['growth_rate']*100:.1f}% rate)")
