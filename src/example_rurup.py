#!/usr/bin/env python3
"""
Example usage of the Rurup (Rürup Rente / Basisrente) calculator
"""

from src.rurup import Rurup


if __name__ == "__main__":
    print("=" * 70)
    print("RÜRUP RENTE (BASISRENTE) CALCULATOR")
    print("=" * 70)
    
    # Example: Self-employed person contributing to Rürup pension
    rurup = Rurup(
        saving_contribution=400,   # €400/month
        horizon=30,                 # 30 years until retirement
        frequency='monthly',
        initial_balance=0,
        tax_refund_rate=0.40       # 40% tax refund (depends on tax bracket)
    )
    
    # Rürup typically has higher TER (1.5% default)
    # You can adjust if needed:
    # rurup.ter = 0.02  # 2% for some providers
    
    result = rurup.calculate_balance()
    
    print(f"\nMonthly contribution: €{rurup.saving_contribution:,.2f}")
    print(f"Investment horizon: {rurup.horizon} years")
    print(f"Tax refund rate: {rurup.tax_refund_rate*100:.0f}%")
    print(f"TER: {rurup.ter*100:.2f}%")
    print(f"Growth rate: {rurup.average_yearly_growth*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total invested: €{result['total_invested']:,.2f}")
    print(f"Cumulative tax refunds: €{result['cumulative_tax_refunds']:,.2f}")
    print(f"Net cost (after refunds): €{result['net_cost']:,.2f}")
    print(f"Final balance: €{result['final_balance']:,.2f}")
    print(f"Total gains: €{result['final_balance'] - result['total_invested']:,.2f}")
    print(f"Effective gain (vs net cost): €{result['final_balance'] - result['net_cost']:,.2f}")
    print(f"\nVorabpauschale (advance tax):")
    print(f"  Total Vorabpauschale: €{result['total_vorabpauschale']:,.2f}")
    print(f"  Total tax paid: €{result['total_tax_paid']:,.2f}")
    
    # Show first 5 and last 5 years
    print("\n" + "=" * 70)
    print("YEARLY DETAILS (First 5 years)")
    print("=" * 70)
    print(f"{'Year':<6} {'Balance':>12} {'Vorabpauschale':>15} {'Tax Paid':>10} {'Tax Refund':>12}")
    print("-" * 70)
    for i, (detail, refund) in enumerate(zip(result['yearly_details'][:5], result['yearly_tax_refunds'][:5])):
        print(f"Year {detail['year']:2d} €{detail['balance_before_tax']:>10,.2f} "
              f"€{detail['vorabpauschale']:>13,.2f} "
              f"€{detail['tax_paid']:>8,.2f} "
              f"€{refund['tax_refund']:>10,.2f}")
    
    print("\n" + "=" * 70)
    print("YEARLY DETAILS (Last 5 years)")
    print("=" * 70)
    print(f"{'Year':<6} {'Balance':>12} {'Vorabpauschale':>15} {'Tax Paid':>10} {'Tax Refund':>12}")
    print("-" * 70)
    for detail, refund in zip(result['yearly_details'][-5:], result['yearly_tax_refunds'][-5:]):
        print(f"Year {detail['year']:2d} €{detail['balance_before_tax']:>10,.2f} "
              f"€{detail['vorabpauschale']:>13,.2f} "
              f"€{detail['tax_paid']:>8,.2f} "
              f"€{refund['tax_refund']:>10,.2f}")
    
    print("\n" + "=" * 70)
    print("COMPARISON WITH LOWER TER")
    print("=" * 70)
    
    # Compare with a lower TER option
    rurup_low_ter = Rurup(
        saving_contribution=400,
        horizon=30,
        frequency='monthly',
        initial_balance=0
    )
    rurup_low_ter.ter = 0.005  # 0.5% TER (like ETF)
    result_low_ter = rurup_low_ter.calculate_balance()
    
    print(f"Standard Rürup (1.5% TER): €{result['final_balance']:,.2f}")
    print(f"Low-cost option (0.5% TER): €{result_low_ter['final_balance']:,.2f}")
    print(f"Difference: €{result_low_ter['final_balance'] - result['final_balance']:,.2f}")
    print(f"Impact of TER: {((result_low_ter['final_balance'] - result['final_balance']) / result['final_balance'] * 100):.1f}% more with lower TER")
    
    print("\n" + "=" * 70)
    print("REINVESTED TAX REFUNDS")
    print("=" * 70)
    
    # Calculate with reinvested tax refunds
    result_with_reinvestment = rurup.calculate_balance_with_reinvestment(
        reinvest_growth_rate=0.07,  # Reinvest in ETF with same 7% growth
        reinvest_ter=0.005  # Low-cost ETF with 0.5% TER
    )
    
    print(f"\nMain Rürup balance: €{result_with_reinvestment['final_balance']:,.2f}")
    print(f"Tax refunds received: €{result_with_reinvestment['cumulative_tax_refunds']:,.2f}")
    print(f"Reinvested refunds balance: €{result_with_reinvestment['reinvested_balance']:,.2f}")
    print(f"Reinvestment gain: €{result_with_reinvestment['reinvestment_gain']:,.2f}")
    print(f"\nTOTAL (Main + Reinvested): €{result_with_reinvestment['total_combined_balance']:,.2f}")
    print(f"Total effective gain: €{result_with_reinvestment['total_combined_balance'] - result_with_reinvestment['net_cost']:,.2f}")
    
    # Show first few years of reinvestment details
    print("\n" + "=" * 70)
    print("REINVESTMENT DETAILS (First 5 years)")
    print("=" * 70)
    print(f"{'Year':<6} {'Refund':>10} {'Years Inv.':>11} {'Final Balance':>15} {'Gain':>12}")
    print("-" * 70)
    reinvest_details = result_with_reinvestment['reinvestment_details']['yearly_reinvestment_details']
    for detail in reinvest_details[:5]:
        print(f"Year {detail['year_received']:2d} €{detail['refund_amount']:>8,.2f} "
              f"{detail['years_invested']:>11d} €{detail['final_balance']:>13,.2f} "
              f"€{detail['gain']:>10,.2f}")
