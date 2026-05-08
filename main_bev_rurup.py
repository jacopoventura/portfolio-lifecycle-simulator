from src.private_saving_plan import PrivateSavingPlan
from src.bev import BeV
from src.rurup import Rurup


def print_comparison_table(result_monthly, result_bev, result_rurup, pension_expectation_years, pension_age):
    """
    Print a comparison table of investment results
    
    Args:
        result_monthly: Result dictionary from PrivateSavingPlan
        result_bev: Result dictionary from BeV
        result_rurup: Result dictionary from Rurup with reinvestment
        pension_expectation_years: Expected years in retirement
        pension_age: Age when pension starts
    """
    print("\n" + "~" * 88)
    print(" " * 29 + "FINAL OVERVIEW - COMPARISON")
    print("~" * 88)
    
    # Column headers
    print(f"\n{'Metric':<22} {'Private':>10} {'BeV':>10} {'Rup(Main)':>10} {'Rup(Tax)':>10} {'Rup(Tot)':>10}")
    print("-" * 88)
    
    # Final Balance
    rurup_total = result_rurup['total_combined_balance']
    print(f"{'Final Balance':.<22} {result_monthly['final_balance']:>10,.0f} {result_bev['final_balance']:>10,.0f} {result_rurup['final_balance']:>10,.0f} {result_rurup['reinvested_balance']:>10,.0f} {rurup_total:>10,.0f}")
    
    # Total Invested
    print(f"{'Total Invested':.<22} {result_monthly['total_invested']:>10,.0f} {result_bev['total_invested']:>10,.0f} {result_rurup['total_invested']:>10,.0f} {result_rurup['cumulative_tax_refunds']:>10,.0f} {result_rurup['net_cost']:>10,.0f}")
    
    # Net Gain
    private_gain = result_monthly['final_balance'] - result_monthly['total_invested']
    bev_gain = result_bev['final_balance'] - result_bev['total_invested']
    rurup_main_gain = result_rurup['final_balance'] - result_rurup['total_invested']
    rurup_reinv_gain = result_rurup['reinvestment_gain']
    rurup_total_gain = result_rurup['total_combined_balance'] - result_rurup['net_cost']
    
    print(f"{'Net Gain':.<22} {private_gain:>10,.0f} {bev_gain:>10,.0f} {rurup_main_gain:>10,.0f} {rurup_reinv_gain:>10,.0f} {rurup_total_gain:>10,.0f}")
    
    # Tax Paid
    private_tax = result_monthly.get('total_tax_paid', 0)
    bev_tax = 0
    rurup_main_tax = 0
    rurup_reinv_tax = result_rurup.get('reinvested_tax_paid', 0)
    rurup_total_tax = rurup_main_tax + rurup_reinv_tax
    
    print(f"{'Tax Unrealized Profit':.<22} {private_tax:>10,.0f} {bev_tax:>10,.0f} {rurup_main_tax:>10,.0f} {rurup_reinv_tax:>10,.0f} {rurup_total_tax:>10,.0f}")
    
    # Return on Investment (%)
    private_roi = (private_gain / result_monthly['total_invested']) * 100
    bev_roi = (bev_gain / result_bev['total_invested']) * 100
    rurup_main_roi = (rurup_main_gain / result_rurup['total_invested']) * 100
    rurup_reinv_roi = (rurup_reinv_gain / result_rurup['cumulative_tax_refunds']) * 100
    rurup_total_roi = (rurup_total_gain / result_rurup['net_cost']) * 100
    
    print(f"{'ROI (%)':.<22} {private_roi:>9.1f}% {bev_roi:>9.1f}% {rurup_main_roi:>9.1f}% {rurup_reinv_roi:>9.1f}% {rurup_total_roi:>9.1f}%")
    
    # Payout Tax Rate
    private_tax_rate = 0.26
    bev_tax_rate = 0.07
    rurup_main_tax_rate = 0.26
    rurup_tax_tax_rate = 0.26  # Same as private
    rurup_total_tax_rate = (result_rurup['final_balance'] * rurup_main_tax_rate + result_rurup['reinvested_balance'] * rurup_tax_tax_rate) / rurup_total
    
    print(f"{'Payout Tax Rate (%)':.<22} {private_tax_rate*100:>9.1f}% {bev_tax_rate*100:>9.1f}% {rurup_main_tax_rate*100:>9.1f}% {rurup_tax_tax_rate*100:>9.1f}% {rurup_total_tax_rate*100:>9.1f}%")
    
    # Expected Tax at Payout
    private_tax_payout = result_monthly['final_balance'] * private_tax_rate
    bev_tax_payout = result_bev['final_balance'] * bev_tax_rate
    rurup_main_tax_payout = result_rurup['final_balance'] * rurup_main_tax_rate
    rurup_tax_tax_payout = result_rurup['reinvested_balance'] * rurup_tax_tax_rate
    rurup_total_tax_payout = rurup_main_tax_payout + rurup_tax_tax_payout
    
    print(f"{'Exp. Tax at Payout':.<22} {private_tax_payout:>10,.0f} {bev_tax_payout:>10,.0f} {rurup_main_tax_payout:>10,.0f} {rurup_tax_tax_payout:>10,.0f} {rurup_total_tax_payout:>10,.0f}")
    
    # Net Payout (Final Balance - Tax at Payout)
    private_net_payout = result_monthly['final_balance'] - private_tax_payout
    bev_net_payout = result_bev['final_balance'] - bev_tax_payout
    rurup_main_net_payout = result_rurup['final_balance'] - rurup_main_tax_payout
    rurup_tax_net_payout = result_rurup['reinvested_balance'] - rurup_tax_tax_payout
    rurup_total_net_payout = rurup_total - rurup_total_tax_payout
    
    print(f"{'Net Payout':.<22} {private_net_payout:>10,.0f} {bev_net_payout:>10,.0f} {rurup_main_net_payout:>10,.0f} {rurup_tax_net_payout:>10,.0f} {rurup_total_net_payout:>10,.0f}")
    
    print("-" * 88)
    
    # Winner
    balances = {
        'Private Plan': result_monthly['final_balance'],
        'BeV': result_bev['final_balance'],
        'Rurup Total': result_rurup['total_combined_balance']
    }
    winner = max(balances, key=balances.get)
    
    print(f"\n🏆 Best Final Balance: \033[1m\033[92m{winner} (€{balances[winner]:,.0f})\033[0m")
    
    # Best Net Payout
    net_payouts = {
        'Private Plan': private_net_payout,
        'BeV': bev_net_payout,
        'Rurup Total': rurup_total_net_payout
    }
    net_payout_winner = max(net_payouts, key=net_payouts.get)
    
    print(f"🏆 Best Net Payout: \033[1m\033[92m{net_payout_winner} (€{net_payouts[net_payout_winner]:,.0f})\033[0m")
    
    print("\nExpected Yearly Pension (Rürup):")
    yearly_pension_brutto = result_rurup.get('expected_yearly_pension_brutto', 0)
    print(f" - Yearly Pension (brutto): €{yearly_pension_brutto:,.0f}")
    
    # Calculate pension from Rurup capital value (main balance only)
    rurup_main_balance = result_rurup['final_balance']
    pension_from_capital = rurup_main_balance / pension_expectation_years
    print(f" - Pension from Rurup capital value with pension expectation: €{pension_from_capital:,.0f}")
    
    # Calculate breakeven pension expectation (main balance only)
    if yearly_pension_brutto > 0:
        breakeven_years = rurup_main_balance / yearly_pension_brutto
        breakeven_age = pension_age + breakeven_years
        print(f" - Breakeven pension expectation (main): {breakeven_years:.1f} years (age {breakeven_age:.0f})")
    
    # Calculate breakeven with Rurup total (main + reinvested)
    if yearly_pension_brutto > 0:
        breakeven_years_total = (rurup_main_balance - rurup_reinv_gain) / yearly_pension_brutto
        breakeven_age_total = pension_age + breakeven_years_total
        print(f" - Breakeven pension expectation (total): {breakeven_years_total:.1f} years (age {breakeven_age_total:.0f})")
    
    print("\nTax at payout:")
    print(f" - Private Plan: 25-27%")
    print(f" - BeV: Residence country tax rules apply, Italy 7%")
    print(f" - Rürup Rente: 20-25%")

    print("~" * 88 + "\n")


if __name__ == "__main__":


    # =============================== SETTINGS ===============================

    # Monthly investment amount
    monthly_amount_brutto = 600  # Gross amount before tax
    monthly_amount_netto = monthly_amount_brutto * (1 - 0.33)  # After tax (33% tax rate)

    # Horizon in years
    age_start_pension = 67
    current_age = 39
    life_expectancy = 82
    total_horizon_years = age_start_pension - current_age
    horizon_to_55_years = 55 - current_age
    pension_expectation_years = life_expectancy - age_start_pension

    # CAGR stock market
    CAGR = 0.07

    # Private investment plan
    plan_monthly = PrivateSavingPlan(saving_contribution=monthly_amount_netto,
                                     horizon=total_horizon_years,
                                     frequency='monthly',
                                     initial_balance=30000)
    plan_monthly.average_yearly_growth = CAGR
    
    # Calculate balance using default horizon (10 years = 120 months)
    result_monthly = plan_monthly.calculate_balance()

    # Bosch BeV
    bev = BeV(
        total_horizon=total_horizon_years,
        horizon_to_55_year_old=horizon_to_55_years,
        initial_balance=22800,  # Initial capital from previous contributions
        yearly_company_contribution=2000,  # €100/month * 12 months
        monthly_contribution=monthly_amount_brutto  # Gross amount before tax
    )
    

    # Rurup rente
    rurup = Rurup(
        saving_contribution=monthly_amount_netto,
        horizon=total_horizon_years,
        frequency='monthly',
        initial_balance=0,
        tax_refund_rate=0.40  # 40% tax refund on contributions
    )
    rurup.average_yearly_growth = CAGR
    # TER is already set to 1.5% by default in Rurup class
    result_rurup = rurup.calculate_balance_with_reinvestment(
        reinvest_growth_rate=CAGR,  # Reinvest in ETF with 7% growth
        reinvest_ter=0.0015  # Low-cost ETF with 0.5% TER
    )
    
    # Calculate BeV balance
    bev.average_yearly_growth = CAGR
    result_bev = bev.calculate_balance()

    # Print parameter settings
    print("\nParameter settings:")
    print(f"Current age: {current_age}")
    print(f"Pension age: {age_start_pension}")
    print(f"Life expectancy: {life_expectancy}")
    print(f"Total horizon (contribution phase): {total_horizon_years} years")
    print(f"Pension expectation (withdrawal phase): {pension_expectation_years} years")
    print(f"Monthly contribution (gross): €{monthly_amount_brutto:.2f}")
    print(f"Monthly contribution (net): €{monthly_amount_netto:.2f}")

    # Print comparison table
    print_comparison_table(result_monthly, result_bev, result_rurup, pension_expectation_years, age_start_pension)

