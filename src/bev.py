"""
Betriebliche Altersversorgung (BeV) Calculator for Bosch Deutschland

Models company pension plans with age-based growth rates and company contributions.
"""


class BeV:
    """
    Calculator for Betriebliche Altersversorgung (company pension) at Bosch Deutschland
    
    Features:
    - Age-based growth rates (5.7% until age 55, then 3.7%)
    - Monthly employee contributions
    - Yearly company contributions
    - Low TER of 0.08%
    """
    
    def __init__(
        self,
        total_horizon: int,
        horizon_to_55_year_old: int,
        initial_balance: float = 0,
        yearly_company_contribution: float = 0,
        monthly_contribution: float = 0,
        ter: float = 0.0008,
        growth_rate_before_55: float = 0.057,
        growth_rate_after_55: float = 0.037
    ):
        """
        Initialize BeV calculator
        
        Args:
            total_horizon: Total investment horizon in years
            horizon_to_55_year_old: Years until age 55
            initial_balance: Starting balance (default: 0)
            yearly_company_contribution: Annual company contribution (default: 0)
            monthly_contribution: Monthly employee contribution (default: 0)
            ter: Total Expense Ratio (default: 0.0008 = 0.08%)
            growth_rate_before_55: Growth rate until age 55 (default: 0.057 = 5.7%)
            growth_rate_after_55: Growth rate after age 55 (default: 0.037 = 3.7%)
        """
        if horizon_to_55_year_old < 0 or total_horizon < 0:
            raise ValueError("Horizons must be non-negative")
        
        if horizon_to_55_year_old > total_horizon:
            raise ValueError("horizon_to_55_year_old cannot exceed total_horizon")
        
        self.total_horizon = total_horizon
        self.horizon_to_55_year_old = horizon_to_55_year_old
        self.initial_balance = initial_balance
        self.yearly_company_contribution = yearly_company_contribution
        self.monthly_contribution = monthly_contribution
        self.ter = ter
        self.growth_rate_before_55 = growth_rate_before_55
        self.growth_rate_after_55 = growth_rate_after_55
        
    @property
    def net_growth_rate_before_55(self):
        """Net growth rate after TER until age 55"""
        return self.growth_rate_before_55 - self.ter
    
    @property
    def net_growth_rate_after_55(self):
        """Net growth rate after TER after age 55"""
        return self.growth_rate_after_55 - self.ter
    
    @property
    def net_monthly_rate_before_55(self):
        """Monthly compound rate until age 55"""
        return (1 + self.net_growth_rate_before_55) ** (1/12) - 1
    
    @property
    def net_monthly_rate_after_55(self):
        """Monthly compound rate after age 55"""
        return (1 + self.net_growth_rate_after_55) ** (1/12) - 1
    
    def calculate_balance(self):
        """
        Calculate final balance with age-based growth rates
        
        Returns:
            dict: Contains final_balance, total_invested, total_employee_contributions,
                  total_company_contributions, and yearly_details
        """
        balance = self.initial_balance
        total_employee_contributions = 0
        total_company_contributions = 0
        yearly_details = []
        
        for year in range(self.total_horizon):
            year_start_balance = balance
            
            # Determine growth rate for this year
            if year < self.horizon_to_55_year_old:
                monthly_rate = self.net_monthly_rate_before_55
                growth_rate = self.net_growth_rate_before_55
            else:
                monthly_rate = self.net_monthly_rate_after_55
                growth_rate = self.net_growth_rate_after_55
            
            # Add yearly company contribution at beginning of year
            balance += self.yearly_company_contribution
            total_company_contributions += self.yearly_company_contribution
            
            # Add monthly contributions and apply monthly growth
            for month in range(12):
                # Add monthly contribution at beginning of month
                balance += self.monthly_contribution
                total_employee_contributions += self.monthly_contribution
                
                # Apply monthly growth
                balance *= (1 + monthly_rate)
            
            year_end_balance = balance
            
            yearly_details.append({
                'year': year + 1,
                'balance_start': year_start_balance,
                'balance_end': year_end_balance,
                'company_contribution': self.yearly_company_contribution,
                'employee_contributions': self.monthly_contribution * 12,
                'growth_rate': growth_rate,
                'gains': year_end_balance - year_start_balance - self.yearly_company_contribution - (self.monthly_contribution * 12)
            })
        
        total_invested = self.initial_balance + total_employee_contributions + total_company_contributions
        
        return {
            'final_balance': balance,
            'total_invested': total_invested,
            'total_employee_contributions': total_employee_contributions,
            'total_company_contributions': total_company_contributions,
            'total_gains': balance - total_invested,
            'yearly_details': yearly_details
        }
