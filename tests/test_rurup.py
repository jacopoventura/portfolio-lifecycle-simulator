import pytest
from src.rurup import Rurup


class TestRurupInitialization:
    """Test initialization of Rürup class"""
    
    def test_init_with_all_parameters(self):
        """Test initialization with all parameters"""
        rurup = Rurup(
            saving_contribution=500,
            horizon=30,
            frequency='monthly',
            initial_balance=1000,
            tax_refund_rate=0.35
        )
        # Set additional properties
        rurup.average_yearly_growth = 0.06
        rurup.ter = 0.02
        rurup.basiszins = 0.0253
        rurup.tax_rate = 0.26375
        rurup.freibetrag = 1000
        
        assert rurup.saving_contribution == 500
        assert rurup.horizon == 30
        assert rurup.frequency == 'monthly'
        assert rurup.initial_balance == 1000
        assert rurup.average_yearly_growth == 0.06
        assert rurup.ter == 0.02
        assert rurup.basiszins == 0.0253
        assert rurup.tax_rate == 0.26375
        assert rurup.freibetrag == 1000
        assert rurup.tax_refund_rate == 0.35
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters"""
        rurup = Rurup(saving_contribution=300, horizon=20)
        assert rurup.ter == 0.015  # Default 1.5% for Rürup (not 0.5% like private plan)
        assert rurup.frequency == 'monthly'
        assert rurup.initial_balance == 0
        assert rurup.average_yearly_growth == 0.07  # Inherited default
        assert rurup.tax_refund_rate == 0.40  # Default 40% tax refund
    
    def test_rurup_higher_default_ter(self):
        """Test that Rürup has higher default TER than private plan"""
        rurup = Rurup(saving_contribution=100, horizon=10)
        # Rürup default TER is 1.5% vs 0.5% for private plan
        assert rurup.ter == 0.015


class TestRurupInheritance:
    """Test that Rürup properly inherits from PrivateSavingPlan"""
    
    def test_calculate_balance_inherited(self):
        """Test that calculate_balance method works via inheritance"""
        rurup = Rurup(
            saving_contribution=100,
            horizon=1,
            frequency='monthly'
        )
        rurup.ter = 0
        rurup.average_yearly_growth = 0
        
        result = rurup.calculate_balance()
        
        # With 0% growth and 0% TER, should equal contributions
        assert 'final_balance' in result
        assert 'total_invested' in result
        assert result['final_balance'] == pytest.approx(1200, rel=1e-5)
    
    def test_properties_inherited(self):
        """Test that properties are inherited correctly"""
        rurup = Rurup(
            saving_contribution=100,
            horizon=10
        )
        rurup.ter = 0.015
        rurup.average_yearly_growth = 0.07
        
        # Test inherited properties
        assert rurup.average_yearly_growth == 0.07
        assert rurup.ter == 0.015
        assert hasattr(rurup, 'basiszins')
        assert hasattr(rurup, 'tax_rate')
        assert hasattr(rurup, 'freibetrag')


class TestRurupCalculations:
    """Test Rürup balance calculations"""
    
    def test_rurup_with_high_ter(self):
        """Test Rürup calculation with typical high TER"""
        rurup = Rurup(
            saving_contribution=200,
            horizon=10,
            frequency='monthly'
        )
        rurup.ter = 0.015  # 1.5%
        rurup.average_yearly_growth = 0.07
        
        result = rurup.calculate_balance()
        
        assert result['final_balance'] > result['total_invested']
        assert result['total_invested'] == 200 * 12 * 10
    
    def test_rurup_monthly_vs_yearly(self):
        """Test that frequency parameter works correctly"""
        rurup_monthly = Rurup(
            saving_contribution=1200,
            horizon=5,
            frequency='monthly'
        )
        rurup_yearly = Rurup(
            saving_contribution=14400,
            horizon=5,
            frequency='yearly'
        )
        
        result_monthly = rurup_monthly.calculate_balance()
        result_yearly = rurup_yearly.calculate_balance()
        
        assert result_monthly['total_invested'] == 1200 * 12 * 5
        assert result_yearly['total_invested'] == 14400 * 5


class TestRurupWithVorabpauschale:
    """Test Rürup Vorabpauschale calculations"""
    
    def test_rurup_vorabpauschale_tracking(self):
        """Test that Vorabpauschale is calculated for Rürup"""
        rurup = Rurup(
            saving_contribution=500,
            horizon=5,
            frequency='monthly'
        )
        rurup.basiszins = 0.0253
        rurup.tax_rate = 0.26375
        rurup.freibetrag = 0
        
        result = rurup.calculate_balance()
        
        # Check that Vorabpauschale fields exist
        assert 'total_vorabpauschale' in result
        assert 'total_tax_paid' in result
        assert 'yearly_details' in result
        
        # With positive growth and basiszins, should have some Vorabpauschale
        assert result['total_vorabpauschale'] > 0
    
    def test_rurup_with_freibetrag(self):
        """Test Rürup with tax-free allowance"""
        rurup = Rurup(
            saving_contribution=100,
            horizon=3,
            frequency='monthly'
        )
        rurup.freibetrag = 1000
        
        result = rurup.calculate_balance()
        
        # Small contributions might stay under Freibetrag
        assert result['total_tax_paid'] >= 0


class TestRurupRealWorldScenario:
    """Test realistic Rürup scenarios"""
    
    def test_typical_rurup_plan(self):
        """Test typical Rürup plan over 30 years"""
        rurup = Rurup(
            saving_contribution=300,
            horizon=30,
            frequency='monthly',
            initial_balance=0
        )
        rurup.ter = 0.015  # 1.5% typical for Rürup
        rurup.average_yearly_growth = 0.06  # More conservative for Rürup
        
        result = rurup.calculate_balance()
        
        # Sanity checks
        assert result['final_balance'] > result['total_invested']
        assert result['total_invested'] == 300 * 12 * 30  # €108,000
        assert result['final_balance'] > 200000  # Should grow significantly
        
        # Check yearly details structure
        assert len(result['yearly_details']) == 30
        for detail in result['yearly_details']:
            assert 'year' in detail
            assert 'vorabpauschale' in detail
            assert 'tax_paid' in detail


class TestRurupTaxRefunds:
    """Test Rürup tax refund calculations"""
    
    def test_tax_refund_property(self):
        """Test that tax refund rate can be set and retrieved"""
        rurup = Rurup(saving_contribution=100, horizon=10)
        assert rurup.tax_refund_rate == 0.40  # Default
        
        rurup.tax_refund_rate = 0.35
        assert rurup.tax_refund_rate == 0.35
    
    def test_cumulative_tax_refunds_monthly(self):
        """Test cumulative tax refunds with monthly contributions"""
        rurup = Rurup(
            saving_contribution=200,
            horizon=5,
            frequency='monthly',
            tax_refund_rate=0.40
        )
        result = rurup.calculate_balance()
        
        # Check that tax refund fields exist
        assert 'cumulative_tax_refunds' in result
        assert 'yearly_tax_refunds' in result
        assert 'net_cost' in result
        
        # Calculate expected values
        yearly_contribution = 200 * 12  # €2,400/year
        expected_yearly_refund = yearly_contribution * 0.40  # €960/year
        expected_cumulative = expected_yearly_refund * 5  # €4,800 total
        
        assert result['cumulative_tax_refunds'] == pytest.approx(expected_cumulative)
        assert len(result['yearly_tax_refunds']) == 5
        
        # Check first year refund
        assert result['yearly_tax_refunds'][0]['year'] == 1
        assert result['yearly_tax_refunds'][0]['yearly_contribution'] == 2400
        assert result['yearly_tax_refunds'][0]['tax_refund'] == pytest.approx(960)
    
    def test_cumulative_tax_refunds_yearly(self):
        """Test cumulative tax refunds with yearly contributions"""
        rurup = Rurup(
            saving_contribution=5000,
            horizon=3,
            frequency='yearly',
            tax_refund_rate=0.35
        )
        result = rurup.calculate_balance()
        
        expected_yearly_refund = 5000 * 0.35  # €1,750/year
        expected_cumulative = expected_yearly_refund * 3  # €5,250 total
        
        assert result['cumulative_tax_refunds'] == pytest.approx(expected_cumulative)
        assert len(result['yearly_tax_refunds']) == 3
    
    def test_net_cost_calculation(self):
        """Test that net cost accounts for tax refunds"""
        rurup = Rurup(
            saving_contribution=300,
            horizon=10,
            frequency='monthly',
            tax_refund_rate=0.40
        )
        result = rurup.calculate_balance()
        
        total_invested = 300 * 12 * 10  # €36,000
        cumulative_refunds = result['cumulative_tax_refunds']
        expected_net_cost = total_invested - cumulative_refunds
        
        assert result['total_invested'] == total_invested
        assert result['net_cost'] == pytest.approx(expected_net_cost)
        assert result['net_cost'] < result['total_invested']
    
    def test_zero_tax_refund_rate(self):
        """Test with zero tax refund rate"""
        rurup = Rurup(
            saving_contribution=100,
            horizon=5,
            frequency='monthly',
            tax_refund_rate=0.0
        )
        result = rurup.calculate_balance()
        
        assert result['cumulative_tax_refunds'] == 0
        assert result['net_cost'] == result['total_invested']
    
    def test_tax_refund_yearly_breakdown(self):
        """Test yearly tax refund breakdown"""
        rurup = Rurup(
            saving_contribution=500,
            horizon=3,
            frequency='monthly',
            tax_refund_rate=0.42
        )
        result = rurup.calculate_balance()
        
        yearly_contribution = 500 * 12  # €6,000/year
        expected_refund = yearly_contribution * 0.42  # €2,520/year
        
        for i, detail in enumerate(result['yearly_tax_refunds']):
            assert detail['year'] == i + 1
            assert detail['yearly_contribution'] == yearly_contribution
            assert detail['tax_refund'] == pytest.approx(expected_refund)
    
    def test_real_world_scenario_with_refunds(self):
        """Test realistic scenario with tax refunds"""
        rurup = Rurup(
            saving_contribution=400,
            horizon=30,
            frequency='monthly',
            tax_refund_rate=0.40
        )
        rurup.ter = 0.015
        rurup.average_yearly_growth = 0.06
        
        result = rurup.calculate_balance()
        
        # Verify all key fields
        assert result['total_invested'] == 400 * 12 * 30  # €144,000
        assert result['cumulative_tax_refunds'] == pytest.approx(400 * 12 * 30 * 0.40)  # €57,600
        assert result['net_cost'] == pytest.approx(144000 - 57600)  # €86,400
        assert result['final_balance'] > result['net_cost']


class TestRurupReinvestment:
    """Test Rürup tax refund reinvestment calculations"""
    
    def test_reinvested_tax_refunds_basic(self):
        """Test basic reinvestment calculation"""
        rurup = Rurup(
            saving_contribution=100,
            horizon=3,
            frequency='monthly',
            tax_refund_rate=0.40
        )
        rurup.ter = 0
        rurup.average_yearly_growth = 0.10  # 10% for easy calculation
        
        result = rurup.calculate_reinvested_tax_refunds(
            reinvest_growth_rate=0.10,
            reinvest_ter=0
        )
        
        # Check structure
        assert 'final_balance_reinvested' in result
        assert 'total_refunds_invested' in result
        assert 'reinvestment_gain' in result
        assert 'yearly_reinvestment_details' in result
        
        # Verify details
        assert len(result['yearly_reinvestment_details']) == 3
        assert result['total_refunds_invested'] == pytest.approx(100 * 12 * 3 * 0.40)  # €1,440
        assert result['final_balance_reinvested'] > result['total_refunds_invested']
    
    def test_reinvested_refunds_yearly_contributions(self):
        """Test reinvestment with yearly contributions"""
        rurup = Rurup(
            saving_contribution=1000,
            horizon=5,
            frequency='yearly',
            tax_refund_rate=0.35
        )
        
        result = rurup.calculate_reinvested_tax_refunds(
            reinvest_growth_rate=0.05,
            reinvest_ter=0
        )
        
        yearly_refund = 1000 * 0.35  # €350/year
        assert result['total_refunds_invested'] == pytest.approx(350 * 5)
        
        # Check that each year has correct remaining years
        details = result['yearly_reinvestment_details']
        assert details[0]['years_invested'] == 4  # Year 1 refund grows for 4 years
        assert details[1]['years_invested'] == 3  # Year 2 refund grows for 3 years
        assert details[2]['years_invested'] == 2
        assert details[3]['years_invested'] == 1
        assert details[4]['years_invested'] == 0  # Last year no growth
    
    def test_reinvested_last_year_no_growth(self):
        """Test that last year's refund doesn't grow"""
        rurup = Rurup(
            saving_contribution=200,
            horizon=3,
            frequency='monthly',
            tax_refund_rate=0.40
        )
        
        result = rurup.calculate_reinvested_tax_refunds(
            reinvest_growth_rate=0.10,
            reinvest_ter=0
        )
        
        last_year = result['yearly_reinvestment_details'][-1]
        assert last_year['year_received'] == 3
        assert last_year['years_invested'] == 0
        assert last_year['final_balance'] == pytest.approx(last_year['refund_amount'])
        assert last_year['gain'] == 0
    
    def test_calculate_balance_with_reinvestment(self):
        """Test combined calculation"""
        rurup = Rurup(
            saving_contribution=300,
            horizon=10,
            frequency='monthly',
            tax_refund_rate=0.40
        )
        rurup.ter = 0.015
        rurup.average_yearly_growth = 0.07
        
        result = rurup.calculate_balance_with_reinvestment(
            reinvest_growth_rate=0.07,
            reinvest_ter=0.005
        )
        
        # Check all fields are present
        assert 'final_balance' in result  # Main Rürup balance
        assert 'reinvested_balance' in result
        assert 'total_combined_balance' in result
        assert 'reinvestment_details' in result
        
        # Verify combined balance
        assert result['total_combined_balance'] == pytest.approx(
            result['final_balance'] + result['reinvested_balance']
        )
        
        # Verify reinvested balance is positive
        assert result['reinvested_balance'] > 0
        assert result['reinvestment_gain'] > 0
    
    def test_reinvestment_with_different_growth_rates(self):
        """Test reinvestment with different growth rate than main plan"""
        rurup = Rurup(
            saving_contribution=200,
            horizon=5,
            frequency='monthly',
            tax_refund_rate=0.40
        )
        rurup.average_yearly_growth = 0.05  # Main plan: 5%
        
        # Reinvest at higher rate (e.g., more aggressive ETF)
        result = rurup.calculate_reinvested_tax_refunds(
            reinvest_growth_rate=0.08,  # Reinvestment: 8%
            reinvest_ter=0.003
        )
        
        assert result['reinvest_growth_rate'] == 0.08
        assert result['reinvest_ter'] == 0.003
        assert result['reinvestment_gain'] > 0
    
    def test_reinvestment_defaults(self):
        """Test that reinvestment uses sensible defaults"""
        rurup = Rurup(
            saving_contribution=100,
            horizon=5,
            frequency='monthly',
            tax_refund_rate=0.40
        )
        rurup.average_yearly_growth = 0.06
        
        # Call without specifying reinvestment parameters
        result = rurup.calculate_reinvested_tax_refunds()
        
        # Should use parent's growth rate and 0.5% TER
        assert result['reinvest_growth_rate'] == 0.06
        assert result['reinvest_ter'] == 0.005
    
    def test_reinvestment_real_world_scenario(self):
        """Test realistic 30-year scenario with reinvestment"""
        rurup = Rurup(
            saving_contribution=400,
            horizon=30,
            frequency='monthly',
            tax_refund_rate=0.40
        )
        rurup.ter = 0.015
        rurup.average_yearly_growth = 0.06
        
        result = rurup.calculate_balance_with_reinvestment(
            reinvest_growth_rate=0.07,  # Slightly better return on ETF
            reinvest_ter=0.005
        )
        
        # Sanity checks
        main_balance = result['final_balance']
        reinvested_balance = result['reinvested_balance']
        combined = result['total_combined_balance']
        
        assert combined == pytest.approx(main_balance + reinvested_balance)
        assert reinvested_balance > result['cumulative_tax_refunds']
        assert combined > main_balance
        
        # Check that reinvestment significantly boosts total
        boost_percentage = (reinvested_balance / main_balance) * 100
        assert boost_percentage > 0  # Should provide meaningful boost
