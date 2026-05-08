import pytest
from src.private_saving_plan import PrivateSavingPlan


class TestPrivateSavingPlanInitialization:
    """Test initialization of PrivateSavingPlan"""
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters"""
        plan = PrivateSavingPlan(saving_contribution=500, horizon=10)
        assert plan.saving_contribution == 500
        assert plan.horizon == 10
        assert plan.frequency == 'monthly'
        assert plan.initial_balance == 0
    
    def test_init_with_all_parameters(self):
        """Test initialization with all parameters"""
        plan = PrivateSavingPlan(
            saving_contribution=1000,
            horizon=20,
            frequency='yearly',
            initial_balance=5000
        )
        assert plan.saving_contribution == 1000
        assert plan.horizon == 20
        assert plan.frequency == 'yearly'
        assert plan.initial_balance == 5000
    
    def test_default_rates(self):
        """Test that default rates are set correctly"""
        plan = PrivateSavingPlan(saving_contribution=500, horizon=10)
        assert plan.average_yearly_growth == 0.07
        assert plan.ter == 0.005
        assert plan.basiszins == 0.0229
        assert plan.tax_rate == 0.26375
        assert plan.freibetrag == 0


class TestPrivateSavingPlanProperties:
    """Test property getters and setters"""
    
    def test_average_yearly_growth_property(self):
        """Test average_yearly_growth getter and setter"""
        plan = PrivateSavingPlan(saving_contribution=500, horizon=10)
        plan.average_yearly_growth = 0.08
        assert plan.average_yearly_growth == 0.08
    
    def test_ter_property(self):
        """Test TER getter and setter"""
        plan = PrivateSavingPlan(saving_contribution=500, horizon=10)
        plan.ter = 0.01
        assert plan.ter == 0.01
    
    def test_basiszins_property(self):
        """Test basiszins getter and setter"""
        plan = PrivateSavingPlan(saving_contribution=500, horizon=10)
        plan.basiszins = 0.03
        assert plan.basiszins == 0.03
    
    def test_tax_rate_property(self):
        """Test tax_rate getter and setter"""
        plan = PrivateSavingPlan(saving_contribution=500, horizon=10)
        plan.tax_rate = 0.30
        assert plan.tax_rate == 0.30
    
    def test_freibetrag_property(self):
        """Test freibetrag getter and setter"""
        plan = PrivateSavingPlan(saving_contribution=500, horizon=10)
        plan.freibetrag = 2000
        assert plan.freibetrag == 2000


class TestMonthlyCalculations:
    """Test monthly investment calculations"""
    
    def test_calculate_balance_monthly_no_initial(self):
        """Test monthly calculation without initial balance"""
        horizon_years = 1
        monthly_contribution = 100
        # Result from investor.gov 
        plan = PrivateSavingPlan(saving_contribution=monthly_contribution, horizon=horizon_years, frequency='monthly')
        result = plan.calculate_balance(months=horizon_years * 12)
        
        assert 'final_balance' in result
        assert 'total_invested' in result
        assert 'total_vorabpauschale' in result
        assert 'total_tax_paid' in result
        assert 'yearly_details' in result
        
        assert result['total_invested'] == 1200  # 100 * 12
        assert result['final_balance'] > result['total_invested']  # Should have growth
    
    def test_calculate_balance_monthly_with_initial(self):
        """Test monthly calculation with initial balance"""
        plan = PrivateSavingPlan(
            saving_contribution=500,
            horizon=10,
            frequency='monthly',
            initial_balance=10000
        )
        result = plan.calculate_balance(months=120)
        
        assert result['total_invested'] == 70000  # 10000 + 500 * 120
        assert result['final_balance'] > result['total_invested']
    
    def test_monthly_zero_months(self):
        """Test monthly calculation with zero months"""
        plan = PrivateSavingPlan(saving_contribution=500, horizon=10, frequency='monthly')
        result = plan.calculate_balance(months=0)
        
        assert result['total_invested'] == 0
        assert result['final_balance'] == 0
    
    def test_monthly_one_year(self):
        """Test monthly calculation for exactly one year"""
        plan = PrivateSavingPlan(saving_contribution=100, horizon=1, frequency='monthly')
        result = plan.calculate_balance(months=12)
        
        assert result['total_invested'] == 1200
        assert len(result['yearly_details']) == 1  # Should have one year of Vorabpauschale


class TestYearlyCalculations:
    """Test yearly investment calculations"""
    
    def test_calculate_balance_yearly_no_initial(self):
        """Test yearly calculation without initial balance"""
        plan = PrivateSavingPlan(saving_contribution=6000, horizon=10, frequency='yearly')
        result = plan.calculate_balance(months=120)
        
        assert 'final_balance' in result
        assert 'total_invested' in result
        assert 'total_vorabpauschale' in result
        assert 'total_tax_paid' in result
        
        assert result['total_invested'] == 60000  # 6000 * 10 years
        assert result['final_balance'] > result['total_invested']
    
    def test_calculate_balance_yearly_with_initial(self):
        """Test yearly calculation with initial balance"""
        plan = PrivateSavingPlan(
            saving_contribution=5000,
            horizon=10,
            frequency='yearly',
            initial_balance=10000
        )
        result = plan.calculate_balance(months=120)
        
        assert result['total_invested'] == 60000  # 10000 + 5000 * 10
        assert result['final_balance'] > result['total_invested']
    
    def test_yearly_partial_years(self):
        """Test yearly calculation with partial years (should only count full years)"""
        plan = PrivateSavingPlan(saving_contribution=1000, horizon=3, frequency='yearly')
        result = plan.calculate_balance(months=30)  # 2.5 years
        
        assert result['total_invested'] == 2000  # Only 2 full years
        assert len(result['yearly_details']) == 2


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_invalid_frequency(self):
        """Test that invalid frequency raises ValueError"""
        plan = PrivateSavingPlan(saving_contribution=500, horizon=10, frequency='quarterly')
        
        with pytest.raises(ValueError, match="Invalid frequency"):
            plan.calculate_balance(months=12)
    
    def test_negative_growth(self):
        """Test calculation with negative growth (market downturn)"""
        plan = PrivateSavingPlan(saving_contribution=500, horizon=10, frequency='monthly')
        plan.average_yearly_growth = -0.05  # -5% annual return
        result = plan.calculate_balance(months=12)
        
        # Should still calculate, but final balance may be less than invested
        assert 'final_balance' in result
        assert result['total_invested'] == 6000
    
    def test_high_ter(self):
        """Test calculation with very high TER"""
        plan = PrivateSavingPlan(saving_contribution=500, horizon=10, frequency='monthly')
        plan.ter = 0.03  # 3% TER (very high)
        result = plan.calculate_balance(months=12)
        
        assert result['final_balance'] > 0
    
    def test_override_initial_balance_in_calculate(self):
        """Test that initial_balance can be overridden in calculate_balance"""
        plan = PrivateSavingPlan(
            saving_contribution=500,
            horizon=10,
            frequency='monthly',
            initial_balance=5000
        )
        result = plan.calculate_balance(months=12, initial_balance=10000)
        
        assert result['total_invested'] == 16000  # 10000 + 500 * 12


class TestRealWorldScenarios:
    """Test with real-world scenarios"""
    
    def test_typical_retirement_plan(self):
        """Test typical retirement savings plan"""
        # Placeholder test for real-world retirement scenario
        plan = PrivateSavingPlan(
            saving_contribution=500,
            horizon=30,
            frequency='monthly',
            initial_balance=0
        )
        result = plan.calculate_balance()
        
        # Basic sanity checks
        assert result['final_balance'] > result['total_invested']
        assert result['total_invested'] == 500 * 30 * 12  # 180,000


