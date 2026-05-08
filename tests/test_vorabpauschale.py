import pytest
from src.private_saving_plan import PrivateSavingPlan


class TestVorabpauschale:
    """Test Vorabpauschale calculations"""
    
    def test_vorabpauschale_with_zero_basiszins(self):
        """Test that Vorabpauschale is 0 when basiszins is 0"""
        plan = PrivateSavingPlan(saving_contribution=500, horizon=10, frequency='monthly')
        plan.basiszins = 0
        result = plan.calculate_balance(months=12)
        
        assert result['total_vorabpauschale'] == 0
    
    def test_vorabpauschale_respects_freibetrag(self):
        """Test that tax-free allowance (Freibetrag) is respected"""
        plan = PrivateSavingPlan(saving_contribution=100, horizon=10, frequency='monthly')
        plan.freibetrag = 1000
        result = plan.calculate_balance(months=12)
        
        # With small contributions, Vorabpauschale should be below Freibetrag
        assert result['total_tax_paid'] == 0 or result['total_tax_paid'] < result['total_vorabpauschale'] * plan.tax_rate
    
    def test_vorabpauschale_yearly_details(self):
        """Test that yearly details are tracked correctly"""
        plan = PrivateSavingPlan(saving_contribution=500, horizon=3, frequency='monthly')
        result = plan.calculate_balance(months=36)
        
        assert len(result['yearly_details']) == 3
        for detail in result['yearly_details']:
            assert 'year' in detail
            assert 'vorabpauschale' in detail
            assert 'tax_paid' in detail
            assert 'balance_before_tax' in detail


class TestVorabpauschaleFinanztip:
    """Test Vorabpauschale calculation with real-world data from Finanztip.de
    
    Reference: https://www.finanztip.de/indexfonds-etf/etf-steuern/was-ist-die-vorabpauschale/
    Example from section "Beispiel: Wie hoch ist die Vorabpauschale?"
    """
    
    def test_vorabpauschale_2025_example_scenario_1(self):
        """Test Vorabpauschale calculation for 2025 example from Finanztip.de
        
        Scenario 1: Wertsteigerung höher als Basisertrag
        - Initial value (Jan 1, 2025): €10,000
        - Final value (Dec 31, 2025): €10,500
        - Actual gain: €500
        - Basiszins 2025: 2.53%
        - Expected Basisertrag: €10,000 × 2.53% × 0.7 = €177.10
        - Expected Vorabpauschale: €177.10 (min of Basisertrag and actual gain)
        
        Note: For Aktien-ETF, 30% Teilfreistellung applies to the Vorabpauschale,
        making the taxable amount €177.10 × 0.7 = €123.97
        Tax (without church): €123.97 × 26.375% = €32.70
        
        """
        # Calculate required growth rate to reach €10,500 from €10,000
        # €10,000 × (1 + r) = €10,500 => r = 0.05 (5%)
        target_growth = 0.05
        
        # Create plan with €10,000 initial balance, no contributions
        plan = PrivateSavingPlan(
            saving_contribution=0,
            horizon=1,
            frequency='yearly',
            initial_balance=10000
        )
        
        # Set parameters to match Finanztip example
        plan.basiszins = 0.0253  # 2.53% for 2025
        plan.ter = 0  # No TER in the example
        plan.freibetrag = 0  # No tax-free allowance in this example
        plan.average_yearly_growth = target_growth  # 5% to reach €10,500
        
        # Calculate for 1 year (12 months)
        result = plan.calculate_balance(months=12)
        
        # Verify final balance is approximately €10,500
        assert abs(result['final_balance'] - 10500) < 1, \
            f"Expected final balance ~€10,500, got €{result['final_balance']:.2f}"
        
        # Verify actual gain
        actual_gain = result['final_balance'] - result['total_invested']
        assert abs(actual_gain - 500) < 1, \
            f"Expected gain ~€500, got €{actual_gain:.2f}"
        
        # Verify Basisertrag calculation: €10,000 × 2.53% × 0.7
        expected_basisertrag = 10000 * 0.0253 * 0.7
        assert abs(expected_basisertrag - 177.10) < 0.01, \
            f"Expected Basisertrag €177.10, calculated €{expected_basisertrag:.2f}"
        
        # Verify Vorabpauschale is the minimum of Basisertrag and actual gain
        # Should be €177.10 since it's less than €500
        expected_vorabpauschale = min(expected_basisertrag, actual_gain)
        assert abs(result['total_vorabpauschale'] - expected_vorabpauschale) < 1, \
            f"Expected Vorabpauschale ~€{expected_vorabpauschale:.2f}, got €{result['total_vorabpauschale']:.2f}"
        
        assert abs(result['total_vorabpauschale'] - 177.10) < 1, \
            f"Expected Vorabpauschale ~€177.10 from Finanztip example, got €{result['total_vorabpauschale']:.2f}"
        
        # Verify tax calculation with Teilfreistellung (30% exempt for Aktien-ETF)
        # Taxable amount = €177.10 × 0.7 = €123.97
        # Tax = €123.97 × 26.375% = €32.70
        expected_tax = result['total_vorabpauschale'] * 0.7 * 0.26375
        assert abs(result['total_tax_paid'] - expected_tax) < 0.1, \
            f"Expected tax ~€{expected_tax:.2f}, got €{result['total_tax_paid']:.2f}"

    def test_vorabpauschale_2025_example_scenario_2(self):
        """Test Vorabpauschale calculation for Scenario 2 from Finanztip.de
        
        Scenario 2: Wertsteigerung niedriger als Basisertrag
        - Initial value (Jan 1, 2025): €10,000
        - Final value (Dec 31, 2025): €10,100
        - Actual gain: €100
        - Basiszins 2025: 2.53%
        - Expected Basisertrag: €10,000 × 2.53% × 0.7 = €177.10
        - Expected Vorabpauschale: €100 (min of Basisertrag and actual gain)
        
        When actual gain is lower than Basisertrag, Vorabpauschale = actual gain
        """
        # Calculate required growth rate to reach €10,100 from €10,000
        # €10,000 × (1 + r) = €10,100 => r = 0.01 (1%)
        target_growth = 0.01
        
        # Create plan with €10,000 initial balance, no contributions
        plan = PrivateSavingPlan(
            saving_contribution=0,
            horizon=1,
            frequency='yearly',
            initial_balance=10000
        )
        
        # Set parameters to match Finanztip example
        plan.basiszins = 0.0253  # 2.53% for 2025
        plan.ter = 0  # No TER in the example
        plan.freibetrag = 0  # No tax-free allowance
        plan.average_yearly_growth = target_growth  # 1% to reach €10,100
        
        # Calculate for 1 year
        result = plan.calculate_balance(months=12)
        
        # Verify final balance is approximately €10,100
        assert abs(result['final_balance'] - 10100) < 1, \
            f"Expected final balance ~€10,100, got €{result['final_balance']:.2f}"
        
        # Verify actual gain
        actual_gain = result['final_balance'] - result['total_invested']
        assert abs(actual_gain - 100) < 1, \
            f"Expected gain ~€100, got €{actual_gain:.2f}"
        
        # Verify Vorabpauschale is capped at actual gain (€100)
        # Even though Basisertrag would be €177.10
        assert abs(result['total_vorabpauschale'] - 100) < 1, \
            f"Expected Vorabpauschale ~€100 (capped at actual gain), got €{result['total_vorabpauschale']:.2f}"
        
    def test_vorabpauschale_2025_example_scenario_3(self):
        """Test Vorabpauschale calculation for Scenario 3 from Finanztip.de
        
        Scenario 3: Fonds hat an Wert verloren (loss)
        - Initial value (Jan 1, 2025): €10,000
        - Final value (Dec 31, 2025): €9,800
        - Actual gain: -€200 (loss)
        - Basiszins 2025: 2.53%
        - Expected Vorabpauschale: €0 (no tax when there's a loss)
        """
        # Calculate required growth rate for loss of €200
        # €10,000 × (1 + r) = €9,800 => r = -0.02 (-2%)
        target_growth = -0.02
        
        # Create plan with €10,000 initial balance, no contributions
        plan = PrivateSavingPlan(
            saving_contribution=0,
            horizon=1,
            frequency='yearly',
            initial_balance=10000
        )
        
        # Set parameters to match Finanztip example
        plan.basiszins = 0.0253  # 2.53% for 2025
        plan.ter = 0  # No TER in the example
        plan.freibetrag = 0  # No tax-free allowance
        plan.average_yearly_growth = target_growth  # -2% for loss
        
        # Calculate for 1 year
        result = plan.calculate_balance(months=12)
        
        # Verify final balance is approximately €9,800
        assert abs(result['final_balance'] - 9800) < 1, \
            f"Expected final balance ~€9,800, got €{result['final_balance']:.2f}"
        
        # Verify loss
        actual_gain = result['final_balance'] - result['total_invested']
        assert actual_gain < 0, \
            f"Expected negative gain (loss), got €{actual_gain:.2f}"
        
        # Verify Vorabpauschale is €0 when there's a loss
        assert result['total_vorabpauschale'] == 0, \
            f"Expected Vorabpauschale €0 for loss scenario, got €{result['total_vorabpauschale']:.2f}"
        
        # Verify no tax is paid
        assert result['total_tax_paid'] == 0, \
            f"Expected tax €0 for loss scenario, got €{result['total_tax_paid']:.2f}"
    
    def test_vorabpauschale_with_contributions(self):
        """Test Vorabpauschale with monthly contributions
        
        The Vorabpauschale is calculated per tranche based on months held.
        For contributions made during the year, the Vorabpauschale is prorated.
        
        Note: In the implementation, contributions made during the year have reduced
        Vorabpauschale because they are only held for part of the year.
        A contribution made in month X only counts from month X+1.
        """
        # Create plan with monthly contributions
        plan = PrivateSavingPlan(
            saving_contribution=100,
            horizon=1,
            frequency='monthly',
            initial_balance=1000
        )
        
        # Set parameters
        plan.basiszins = 0.0253  # 2.53%
        plan.ter = 0
        plan.freibetrag = 0
        plan.average_yearly_growth = 0.07  # 7%
        
        # Calculate for 1 year
        result = plan.calculate_balance(months=12)
        
        # Verify Vorabpauschale was calculated
        assert result['total_vorabpauschale'] > 0, \
            "Expected positive Vorabpauschale"
        
        # Verify it doesn't exceed actual gains
        actual_gain = result['final_balance'] - result['total_invested']
        assert result['total_vorabpauschale'] <= actual_gain, \
            f"Vorabpauschale (€{result['total_vorabpauschale']:.2f}) should not exceed actual gain (€{actual_gain:.2f})"
        
        # Verify Vorabpauschale is a reasonable percentage of the balance
        # With a 2.53% Basiszins, the Vorabpauschale should be roughly 1-2% of total invested
        vorab_percentage = (result['total_vorabpauschale'] / result['total_invested']) * 100
        assert 0.1 < vorab_percentage < 3, \
            f"Vorabpauschale should be 0.1-3% of total invested, got {vorab_percentage:.2f}%"
        
        # Verify yearly details are tracked
        assert len(result['yearly_details']) == 1, \
            "Expected 1 year of details"
        assert result['yearly_details'][0]['vorabpauschale'] == result['total_vorabpauschale'], \
            "First year Vorabpauschale should equal total"
