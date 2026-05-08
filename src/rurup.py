"""
Rürup Rente (Basisrente) Calculator

Models Rürup pension plans built on private saving plan with additional tax features.
"""

from src.private_saving_plan import PrivateSavingPlan


class Rurup(PrivateSavingPlan):
    """
    Calculator for Rürup Rente (Basisrente)
    
    Inherits from PrivateSavingPlan and extends it with Rürup-specific features:
    - Typically higher TER (1-2%)
    - Tax advantages during contribution phase
    - Special withdrawal rules
    
    Additional functionality to be implemented as needed.
    """
    
    def __init__(
        self,
        saving_contribution: float,
        horizon: int,
        frequency: str = 'monthly',
        initial_balance: float = 0,
        tax_refund_rate: float = 0.40
    ):
        """
        Initialize Rürup calculator
        
        Args:
            saving_contribution: Amount to save per period
            horizon: Investment horizon in years
            frequency: 'monthly' or 'yearly' (default: 'monthly')
            initial_balance: Starting balance (default: 0)
            tax_refund_rate: Tax refund rate on contributions (default: 0.40 = 40%)
        
        Note: Use property setters to configure:
        - ter: Total Expense Ratio (typically 1-2% for Rürup)
        - average_yearly_growth: Expected annual return
        - basiszins: German tax base rate
        - tax_rate: German capital gains tax
        - freibetrag: Tax-free allowance
        """
        super().__init__(
            saving_contribution=saving_contribution,
            horizon=horizon,
            frequency=frequency,
            initial_balance=initial_balance
        )

        # Interpolation data for expected yearly pension
        self.__month_investment = [200, 300, 500, 1000]
        self.__yearly_pension_brutto = [6337, 9500, 15800, 31700]
        
        # Set Rürup-specific default TER (higher than standard private plan)
        self.ter = 0.015  # 1.5% default for Rürup
        
        # Tax refund rate (percentage of contributions refunded by tax authority)
        self.__tax_refund_rate = tax_refund_rate
    
    @property
    def tax_refund_rate(self):
        """Get the tax refund rate on contributions"""
        return self.__tax_refund_rate
    
    @tax_refund_rate.setter
    def tax_refund_rate(self, rate):
        """Set the tax refund rate on contributions"""
        self.__tax_refund_rate = rate
    
    def calculate_expected_yearly_pension(self, monthly_contribution: float = None) -> float:
        """
        Calculate expected yearly pension (brutto) using linear interpolation
        
        Args:
            monthly_contribution: Monthly contribution amount (default: uses self.saving_contribution)
        
        Returns:
            float: Expected yearly pension (brutto) based on interpolation
        
        Note:
            Uses linear interpolation between known data points:
            - 200€/month → 6,337€/year
            - 500€/month → 15,800€/year
            - 1,000€/month → 31,700€/year
        """
        if monthly_contribution is None:
            if self.frequency == 'monthly':
                monthly_contribution = self.saving_contribution
            else:
                # For yearly contributions, convert to monthly equivalent
                monthly_contribution = self.saving_contribution / 12
        
        # If contribution is outside the range, use linear extrapolation
        if monthly_contribution <= self.__month_investment[0]:
            # Below minimum, use ratio from first data point
            return (monthly_contribution / self.__month_investment[0]) * self.__yearly_pension_brutto[0]
        elif monthly_contribution >= self.__month_investment[-1]:
            # Above maximum, use ratio from last data point
            return (monthly_contribution / self.__month_investment[-1]) * self.__yearly_pension_brutto[-1]
        
        # Find the two points to interpolate between
        for i in range(len(self.__month_investment) - 1):
            if self.__month_investment[i] <= monthly_contribution <= self.__month_investment[i + 1]:
                # Linear interpolation
                x0, x1 = self.__month_investment[i], self.__month_investment[i + 1]
                y0, y1 = self.__yearly_pension_brutto[i], self.__yearly_pension_brutto[i + 1]
                
                # y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
                yearly_pension = y0 + (monthly_contribution - x0) * (y1 - y0) / (x1 - x0)
                return yearly_pension
        
        # Should never reach here
        return 0.0
    
    def calculate_balance(self, months=None, initial_balance=None):
        """
        Calculate balance with Rürup-specific tax refund benefits
        
        Extends parent's calculate_balance by adding:
        - Yearly tax refund calculations based on contributions
        - Cumulative tax refund tracking
        
        Args:
            months: Number of months to calculate (default uses self.horizon * 12)
            initial_balance: Starting balance (default uses self.initial_balance)
        
        Returns:
            dict: Enhanced result with tax refund information:
                - All fields from parent's calculate_balance
                - cumulative_tax_refunds: Total tax refunds received
                - yearly_tax_refunds: List of yearly tax refund amounts
        """
        # Get base calculation from parent
        result = super().calculate_balance(months, initial_balance)
        
        # Calculate yearly contributions
        if self.frequency == 'monthly':
            yearly_contribution = self.saving_contribution * 12
        else:
            yearly_contribution = self.saving_contribution
        
        # Calculate tax refunds for each year
        years = self.horizon if months is None else months // 12
        yearly_tax_refunds = []
        cumulative_tax_refunds = 0
        
        for year in range(years):
            # Tax refund is received after each year of contributions
            tax_refund = yearly_contribution * self.__tax_refund_rate
            yearly_tax_refunds.append({
                'year': year + 1,
                'yearly_contribution': yearly_contribution,
                'tax_refund': tax_refund
            })
            cumulative_tax_refunds += tax_refund
        
        # Add tax refund information to result
        result['cumulative_tax_refunds'] = cumulative_tax_refunds
        result['yearly_tax_refunds'] = yearly_tax_refunds
        result['net_cost'] = result['total_invested'] - cumulative_tax_refunds
        
        # Calculate expected yearly pension
        result['expected_yearly_pension_brutto'] = self.calculate_expected_yearly_pension()
        
        return result
    
    def calculate_reinvested_tax_refunds(
        self,
        reinvest_growth_rate: float = None,
        reinvest_ter: float = None
    ):
        """
        Calculate the final balance of tax refunds reinvested in a private saving plan
        
        Each year's tax refund is reinvested with a horizon of (total_years - year_received).
        For example, in a 28-year plan:
        - Year 1 refund: invested for 27 years
        - Year 2 refund: invested for 26 years
        - ...
        - Year 27 refund: invested for 1 year
        - Year 28 refund: no growth (0 years)
        
        Args:
            reinvest_growth_rate: Growth rate for reinvested funds (default: uses parent's growth rate)
            reinvest_ter: TER for reinvested funds (default: 0.005 = 0.5%, typical for ETF)
        
        Returns:
            dict: Contains:
                - final_balance_reinvested: Total balance from all reinvested refunds
                - total_refunds_invested: Sum of all tax refunds (same as cumulative_tax_refunds)
                - reinvestment_gain: Growth from reinvesting (final_balance - total_refunds)
                - yearly_reinvestment_details: List of per-year details
        """
        # Get the tax refunds first
        result = self.calculate_balance()
        yearly_refunds = result['yearly_tax_refunds']
        
        # Set defaults for reinvestment parameters
        if reinvest_growth_rate is None:
            reinvest_growth_rate = self.average_yearly_growth
        if reinvest_ter is None:
            reinvest_ter = 0.005  # 0.5% - typical for ETF
        
        total_years = self.horizon
        yearly_details = []
        total_final_balance = 0
        total_vorabpauschale = 0
        total_tax_paid = 0
        
        for refund_info in yearly_refunds:
            year_received = refund_info['year']
            refund_amount = refund_info['tax_refund']
            
            # Calculate remaining years for this refund to grow
            # Refund received after year N, so it grows for (total_years - N) years
            remaining_years = total_years - year_received
            
            if remaining_years > 0:
                # Create a private plan for this refund
                refund_plan = PrivateSavingPlan(
                    saving_contribution=0,  # No ongoing contributions
                    horizon=remaining_years,
                    frequency='yearly',
                    initial_balance=refund_amount
                )
                refund_plan.average_yearly_growth = reinvest_growth_rate
                refund_plan.ter = reinvest_ter
                refund_plan.basiszins = self.basiszins  # Calculate Vorabpauschale for reinvested refunds
                refund_plan.tax_rate = self.tax_rate
                refund_plan.freibetrag = 0  # No freibetrag for this separate calculation
                
                # Calculate final balance for this refund
                refund_result = refund_plan.calculate_balance()
                final_balance = refund_result['final_balance']
                vorabpauschale = refund_result.get('total_vorabpauschale', 0)
                tax_paid = refund_result.get('total_tax_paid', 0)
            else:
                # Last year's refund has no time to grow
                final_balance = refund_amount
                vorabpauschale = 0
                tax_paid = 0
            
            yearly_details.append({
                'year_received': year_received,
                'refund_amount': refund_amount,
                'years_invested': remaining_years,
                'final_balance': final_balance,
                'gain': final_balance - refund_amount,
                'vorabpauschale': vorabpauschale,
                'tax_paid': tax_paid
            })
            
            total_final_balance += final_balance
            total_vorabpauschale += vorabpauschale
            total_tax_paid += tax_paid
        
        return {
            'final_balance_reinvested': total_final_balance,
            'total_refunds_invested': result['cumulative_tax_refunds'],
            'reinvestment_gain': total_final_balance - result['cumulative_tax_refunds'],
            'total_vorabpauschale': total_vorabpauschale,
            'total_tax_paid': total_tax_paid,
            'yearly_reinvestment_details': yearly_details,
            'reinvest_growth_rate': reinvest_growth_rate,
            'reinvest_ter': reinvest_ter
        }
    
    def calculate_balance_with_reinvestment(
        self,
        reinvest_growth_rate: float = None,
        reinvest_ter: float = None
    ):
        """
        Calculate complete Rürup balance including reinvested tax refunds
        
        This combines:
        1. The main Rürup balance
        2. The balance from reinvesting tax refunds
        
        Args:
            reinvest_growth_rate: Growth rate for reinvested funds (default: uses parent's growth rate)
            reinvest_ter: TER for reinvested funds (default: 0.005 = 0.5%)
        
        Returns:
            dict: Contains all fields from calculate_balance() plus:
                - reinvested_balance: Balance from reinvested tax refunds
                - total_combined_balance: Main Rürup + reinvested refunds
                - reinvestment_details: Details of the reinvestment calculation
        """
        # Get main Rürup calculation
        main_result = self.calculate_balance()
        
        # Get reinvestment calculation
        reinvest_result = self.calculate_reinvested_tax_refunds(
            reinvest_growth_rate=reinvest_growth_rate,
            reinvest_ter=reinvest_ter
        )
        
        # Combine results
        combined_result = main_result.copy()
        combined_result['reinvested_balance'] = reinvest_result['final_balance_reinvested']
        combined_result['reinvestment_gain'] = reinvest_result['reinvestment_gain']
        combined_result['reinvested_vorabpauschale'] = reinvest_result['total_vorabpauschale']
        combined_result['reinvested_tax_paid'] = reinvest_result['total_tax_paid']
        combined_result['total_combined_balance'] = (
            main_result['final_balance'] + reinvest_result['final_balance_reinvested']
        )
        combined_result['reinvestment_details'] = reinvest_result
        
        return combined_result
