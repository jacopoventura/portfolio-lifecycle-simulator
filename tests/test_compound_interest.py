import pytest
from src.private_saving_plan import PrivateSavingPlan


class TestCompoundInterest:
    """Test compound interest calculations"""
    
    def test_compound_interest_monthly(self):
        """Test that compound interest is applied correctly for monthly contributions"""
        saving_contribution = 100
        
        # TEST 1: 1 year horizon with monthly contributions of €100, 7% growth, 0% TER, 0% tax
        # Benchmark from investor.gov compound interest calculator: https://www.investor.gov/financial-tools-calculators/calculators/compound-interest-calculator
        horizon_years = 1
        initial_balance = 0
        total_invested = initial_balance + saving_contribution * horizon_years * 12
        expected_final_balance = 1239  

        plan_monthly = PrivateSavingPlan(
            saving_contribution=saving_contribution,
            horizon=horizon_years,
            frequency='monthly'
        )
        plan_monthly.ter = 0

        result_monthly = plan_monthly.calculate_balance()
        assert result_monthly['total_invested'] == total_invested
        assert result_monthly['final_balance'] > total_invested
        assert result_monthly['total_invested'] == total_invested
        assert abs(result_monthly['final_balance'] - expected_final_balance) < 1  # Allow small rounding error
        
        # TEST 2: 1 year with initial balance of €1000, no monthly contribution
        initial_balance = 1000
        expected_final_balance = 1070  # 1000 * (1 + 0.07)
        plan_initial = PrivateSavingPlan(
            saving_contribution=0,
            horizon=horizon_years,
            frequency='monthly',
            initial_balance=initial_balance
        )
        plan_initial.ter = 0
        result_initial = plan_initial.calculate_balance()
        assert result_initial['total_invested'] == initial_balance
        assert result_initial['final_balance'] > initial_balance
        assert abs(result_initial['final_balance'] - expected_final_balance) < 1  # Allow small rounding error

        # TEST 3: 1 year with initial balance of €1000 and monthly contributions of €100
        monthly_contribution = 100
        # Expected = FV of initial balance + FV of monthly contributions
        # From TEST 2, we know €1000 grows to €1070
        # From TEST 1, we know €100/month for 1 year grows to €1239
        # Combined: €1070 + (€1239 - €1200) = €1070 + €39 = €1109 (approximately)
        # More precisely: initial grows to ~€1070, contributions add ~€1238 = €2308
        expected_final_balance = 1070 + 1238  # Approximate sum
        plan_combined = PrivateSavingPlan(
            saving_contribution=monthly_contribution,
            horizon=horizon_years,
            frequency='monthly',
            initial_balance=initial_balance
        )
        plan_combined.ter = 0
        result_combined = plan_combined.calculate_balance()
        assert result_combined['total_invested'] == initial_balance + monthly_contribution * horizon_years * 12
        assert abs(result_combined['final_balance'] - expected_final_balance) < 5  # Allow small rounding error


        # TEST 4: 10 year horizon with monthly contributions of €100, 7% growth, 0% TER, 0% tax
        horizon_years = 10
        expected_final_balance = 17105  # Using compound monthly rate: (1.07)^(1/12) - 1
        total_invested = saving_contribution * horizon_years * 12 
        result_monthly = plan_monthly.calculate_balance(months=horizon_years * 12)
        assert result_monthly['total_invested'] == total_invested
        assert result_monthly['final_balance'] > total_invested
        assert abs(result_monthly['final_balance'] - expected_final_balance) < 5  # Allow small rounding error
        
        # TEST 5: 10 year horizon with initial balance of €1000, no monthly contribution
        initial_balance = 1000
        expected_final_balance = 1000 * (1.07) ** horizon_years  # 1000 grows to ~€1967
        plan_initial = PrivateSavingPlan(
            saving_contribution=0,
            horizon=horizon_years,
            frequency='monthly',
            initial_balance=initial_balance
        )
        plan_initial.ter = 0
        result_initial = plan_initial.calculate_balance(months=horizon_years * 12)
        assert result_initial['total_invested'] == initial_balance
        assert result_initial['final_balance'] > initial_balance
        assert abs(result_initial['final_balance'] - expected_final_balance) < 10  # Allow small rounding error

        # TEST 6: 10 year horizon with initial balance of €1000 and monthly contributions of €100
        monthly_contribution = 100
        # Expected = FV of initial balance + FV of monthly contributions
        # From TEST 5, we know €1000 grows to ~€1967
        # From TEST 4, we know €100/month for 10 years grows to ~€17105
        # Combined: €1967 + €17105 = €19072 (approximately)
        expected_final_balance = 1967 + 17105  # Approximate sum
        plan_combined = PrivateSavingPlan(
            saving_contribution=monthly_contribution,
            horizon=horizon_years,
            frequency='monthly',
            initial_balance=initial_balance
        )
        plan_combined.ter = 0
        result_combined = plan_combined.calculate_balance(months=horizon_years * 12)
        assert result_combined['total_invested'] == initial_balance + monthly_contribution * horizon_years * 12
        assert abs(result_combined['final_balance'] - expected_final_balance) < 20  # Allow small rounding error

    def test_compound_interest_yearly(self):
        """Test that compound interest is applied correctly for yearly contributions"""
        
        # TEST 1: 1 year horizon with yearly contributions of €1200, 7% growth, 0% TER, 0% tax
        horizon_years = 1
        yearly_contribution = 1200
        initial_balance = 0
        total_invested = initial_balance + yearly_contribution * horizon_years
        expected_final_balance = 1284  # 1200 * (1 + 0.07)
        plan_yearly = PrivateSavingPlan(
            saving_contribution=yearly_contribution,
            horizon=horizon_years,
            frequency='yearly'
        )
        plan_yearly.ter = 0
        result_yearly = plan_yearly.calculate_balance(months=horizon_years * 12)
        assert result_yearly['total_invested'] == total_invested
        assert result_yearly['final_balance'] > total_invested
        assert abs(result_yearly['final_balance'] - expected_final_balance) < 1

        # TEST 2: 1 year with initial balance of €1000, no yearly contribution
        initial_balance = 1000
        expected_final_balance = 1070  # 1000 * (1 + 0.07)
        plan_initial = PrivateSavingPlan(
            saving_contribution=0,
            horizon=horizon_years,
            frequency='yearly',
            initial_balance=initial_balance
        )
        plan_initial.ter = 0
        result_initial = plan_initial.calculate_balance(months=horizon_years * 12)
        assert result_initial['total_invested'] == initial_balance
        assert result_initial['final_balance'] > initial_balance
        assert abs(result_initial['final_balance'] - expected_final_balance) < 1

        # TEST 3: 1 year with initial balance of €1000 and yearly contributions of €1200
        yearly_contribution = 1200
        # Expected = FV of initial balance + FV of yearly contribution
        # From TEST 2, we know €1000 grows to €1070
        # From TEST 1, we know €1200 grows to €1284
        # Combined: €1070 + €1284 = €2354 (approximately)
        expected_final_balance = 1070 + 1284  # Approximate sum
        plan_combined = PrivateSavingPlan(
            saving_contribution=yearly_contribution,
            horizon=horizon_years,
            frequency='yearly',
            initial_balance=initial_balance
        )
        plan_combined.ter = 0
        result_combined = plan_combined.calculate_balance(months=horizon_years * 12)
        assert result_combined['total_invested'] == initial_balance + yearly_contribution * horizon_years
        assert abs(result_combined['final_balance'] - expected_final_balance) < 5  # Allow small rounding error 

        # TEST 4: 10 year horizon with yearly contributions of €1200, 7% growth, 0% TER, 0% tax
        horizon_years = 10
        yearly_contribution = 1200
        expected_final_balance = 17740  # Yearly contributions compounded at 7%
        total_invested = yearly_contribution * horizon_years
        plan_combined = PrivateSavingPlan(
            saving_contribution=yearly_contribution,
            horizon=horizon_years,
            frequency='yearly',
            initial_balance=0
        )
        plan_combined.ter = 0
        result_yearly = plan_yearly.calculate_balance(months=horizon_years * 12)
        assert result_yearly['total_invested'] == total_invested
        assert result_yearly['final_balance'] > total_invested
        assert abs(result_yearly['final_balance'] - expected_final_balance) < 20

        # TEST 5: 10 year horizon with initial balance of €1000, no yearly contribution
        initial_balance = 1000
        expected_final_balance = 1000 * (1.07) ** horizon_years  # 1000 grows to ~€1967
        plan_initial = PrivateSavingPlan(
            saving_contribution=0,
            horizon=horizon_years,
            frequency='yearly',
            initial_balance=initial_balance
        )
        plan_initial.ter = 0
        result_initial = plan_initial.calculate_balance(months=horizon_years * 12)
        assert result_initial['total_invested'] == initial_balance
        assert result_initial['final_balance'] > initial_balance
        assert abs(result_initial['final_balance'] - expected_final_balance) < 10

        # TEST 6: 10 year horizon with initial balance of €1000 and yearly contributions of €1200
        yearly_contribution = 1200
        # Expected = FV of initial balance + FV of yearly contributions
        # From TEST 5, we know €1000 grows to ~€1967
        # From TEST 4, we know €1200/year for 10 years grows to ~€17740
        # Combined: €1967 + €17740 = €19707 (approximately)
        expected_final_balance = 1967 + 17740  # Approximate sum
        plan_combined = PrivateSavingPlan(
            saving_contribution=yearly_contribution,
            horizon=horizon_years,
            frequency='yearly',
            initial_balance=initial_balance
        )
        plan_combined.ter = 0
        result_combined = plan_combined.calculate_balance(months=horizon_years * 12)
        assert result_combined['total_invested'] == initial_balance + yearly_contribution * horizon_years
        assert abs(result_combined['final_balance'] - expected_final_balance) < 30  # Allow small rounding error
