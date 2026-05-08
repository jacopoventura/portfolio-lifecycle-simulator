import pytest
from src.bev import BeV


class TestBeVInitialization:
    """Test initialization of BeV class"""
    
    def test_init_with_all_parameters(self):
        """Test initialization with all parameters"""
        bev = BeV(
            total_horizon=30,
            horizon_to_55_year_old=20,
            initial_balance=10000,
            yearly_company_contribution=1000,
            monthly_contribution=200
        )
        assert bev.total_horizon == 30
        assert bev.horizon_to_55_year_old == 20
        assert bev.initial_balance == 10000
        assert bev.yearly_company_contribution == 1000
        assert bev.monthly_contribution == 200
        assert bev.ter == 0.0008  # Default
        assert bev.growth_rate_before_55 == 0.057  # Default
        assert bev.growth_rate_after_55 == 0.037  # Default
    
    def test_init_with_defaults(self):
        """Test initialization with minimal parameters"""
        bev = BeV(total_horizon=10, horizon_to_55_year_old=5)
        assert bev.initial_balance == 0
        assert bev.yearly_company_contribution == 0
        assert bev.monthly_contribution == 0
    
    def test_init_invalid_horizon(self):
        """Test that horizon_to_55 cannot exceed total_horizon"""
        with pytest.raises(ValueError, match="cannot exceed total_horizon"):
            BeV(total_horizon=10, horizon_to_55_year_old=15)
    
    def test_init_negative_horizon(self):
        """Test that horizons must be non-negative"""
        with pytest.raises(ValueError, match="must be non-negative"):
            BeV(total_horizon=-5, horizon_to_55_year_old=0)


class TestBeVProperties:
    """Test BeV property calculations"""
    
    def test_net_growth_rates(self):
        """Test net growth rate calculations"""
        bev = BeV(
            total_horizon=10,
            horizon_to_55_year_old=5,
            ter=0.001,
            growth_rate_before_55=0.06,
            growth_rate_after_55=0.04
        )
        assert bev.net_growth_rate_before_55 == pytest.approx(0.059)
        assert bev.net_growth_rate_after_55 == pytest.approx(0.039)
    
    def test_monthly_rates(self):
        """Test monthly compound rate calculations"""
        bev = BeV(total_horizon=10, horizon_to_55_year_old=5)
        # With default 5.7% - 0.08% = 5.62% annual
        expected_monthly_before = (1 + 0.0562) ** (1/12) - 1
        assert bev.net_monthly_rate_before_55 == pytest.approx(expected_monthly_before)
        
        # With default 3.7% - 0.08% = 3.62% annual
        expected_monthly_after = (1 + 0.0362) ** (1/12) - 1
        assert bev.net_monthly_rate_after_55 == pytest.approx(expected_monthly_after)


class TestBeVCalculations:
    """Test BeV balance calculations"""
    
    def test_calculate_balance_only_initial(self):
        """Test calculation with only initial balance"""
        bev = BeV(
            total_horizon=1,
            horizon_to_55_year_old=1,
            initial_balance=10000,
            yearly_company_contribution=0,
            monthly_contribution=0,
            growth_rate_before_55=0.06,
            ter=0
        )
        result = bev.calculate_balance()
        
        # Should grow by 6% over one year with monthly compounding
        expected = 10000 * (1.06 ** (1/12)) ** 12
        assert result['final_balance'] == pytest.approx(expected, rel=1e-5)
        assert result['total_invested'] == 10000
        assert result['total_employee_contributions'] == 0
        assert result['total_company_contributions'] == 0
    
    def test_calculate_balance_with_contributions(self):
        """Test calculation with contributions"""
        bev = BeV(
            total_horizon=1,
            horizon_to_55_year_old=1,
            initial_balance=0,
            yearly_company_contribution=1200,
            monthly_contribution=100,
            growth_rate_before_55=0,
            ter=0
        )
        result = bev.calculate_balance()
        
        # With 0% growth: 1200 (yearly) + 100*12 (monthly) = 2400
        assert result['final_balance'] == pytest.approx(2400)
        assert result['total_invested'] == 2400
        assert result['total_employee_contributions'] == 1200
        assert result['total_company_contributions'] == 1200
    
    def test_growth_rate_transition_at_age_55(self):
        """Test that growth rate changes at age 55"""
        bev = BeV(
            total_horizon=10,
            horizon_to_55_year_old=5,
            initial_balance=10000,
            yearly_company_contribution=0,
            monthly_contribution=0,
            growth_rate_before_55=0.10,
            growth_rate_after_55=0.02,
            ter=0
        )
        result = bev.calculate_balance()
        
        # Check yearly details show different growth rates
        year_5_detail = result['yearly_details'][4]  # Last year before 55
        year_6_detail = result['yearly_details'][5]  # First year after 55
        
        assert year_5_detail['growth_rate'] == pytest.approx(0.10)
        assert year_6_detail['growth_rate'] == pytest.approx(0.02)
    
    def test_yearly_details_structure(self):
        """Test that yearly details contain correct information"""
        bev = BeV(
            total_horizon=2,
            horizon_to_55_year_old=1,
            initial_balance=1000,
            yearly_company_contribution=500,
            monthly_contribution=100
        )
        result = bev.calculate_balance()
        
        assert len(result['yearly_details']) == 2
        for detail in result['yearly_details']:
            assert 'year' in detail
            assert 'balance_start' in detail
            assert 'balance_end' in detail
            assert 'company_contribution' in detail
            assert 'employee_contributions' in detail
            assert 'growth_rate' in detail
            assert 'gains' in detail
            assert detail['employee_contributions'] == 1200  # 12 * 100


class TestBeVEdgeCases:
    """Test edge cases"""
    
    def test_zero_horizon(self):
        """Test with zero horizon"""
        bev = BeV(
            total_horizon=0,
            horizon_to_55_year_old=0,
            initial_balance=1000
        )
        result = bev.calculate_balance()
        
        assert result['final_balance'] == 1000
        assert len(result['yearly_details']) == 0
    
    def test_all_years_after_55(self):
        """Test when already past age 55"""
        bev = BeV(
            total_horizon=10,
            horizon_to_55_year_old=0,  # Already 55
            initial_balance=10000,
            growth_rate_before_55=0.10,
            growth_rate_after_55=0.03,
            ter=0
        )
        result = bev.calculate_balance()
        
        # All years should use after-55 rate
        for detail in result['yearly_details']:
            assert detail['growth_rate'] == pytest.approx(0.03)
    
    def test_high_ter(self):
        """Test with TER higher than growth rate"""
        bev = BeV(
            total_horizon=5,
            horizon_to_55_year_old=5,
            initial_balance=10000,
            growth_rate_before_55=0.02,
            ter=0.03  # TER > growth rate
        )
        result = bev.calculate_balance()
        
        # Should result in negative growth
        assert result['final_balance'] < 10000


class TestBeVRealWorldScenario:
    """Test realistic Bosch BeV scenarios"""
    
    def test_typical_employee_scenario(self):
        """Test typical 30-year-old employee until retirement"""
        bev = BeV(
            total_horizon=37,  # Age 30 to 67
            horizon_to_55_year_old=25,
            initial_balance=5000,
            yearly_company_contribution=1200,
            monthly_contribution=300,
            ter=0.0008,
            growth_rate_before_55=0.057,
            growth_rate_after_55=0.037
        )
        result = bev.calculate_balance()
        
        # Verify basic sanity checks
        assert result['final_balance'] > result['total_invested']
        assert result['total_employee_contributions'] == 300 * 12 * 37
        assert result['total_company_contributions'] == 1200 * 37
        assert result['total_invested'] == 5000 + (300 * 12 * 37) + (1200 * 37)
        assert result['total_gains'] == pytest.approx(
            result['final_balance'] - result['total_invested']
        )
        
        # Verify growth rates switch at year 25
        for i, detail in enumerate(result['yearly_details']):
            if i < 25:
                expected_rate = 0.057 - 0.0008
            else:
                expected_rate = 0.037 - 0.0008
            assert detail['growth_rate'] == pytest.approx(expected_rate)
