def calculate_months_held_in_year(purchase_month: int, current_month: int) -> int:
    current_year = current_month // 12
    purchase_year = purchase_month // 12
    if purchase_year < current_year:
        return 12
    elif purchase_year == current_year:
        month_in_year = purchase_month % 12
        return 12 - (month_in_year + 1)
    return 0


def calculate_yearly_vorabpauschale(
    tranches: list,
    current_month: int,
    balance: float,
    total_invested: float,
    basiszins: float,
) -> float:
    if basiszins <= 0:
        return 0.0
    vorabpauschale_sum = 0.0
    for tranche in tranches:
        months_held = calculate_months_held_in_year(tranche['purchase_month'], current_month)
        if months_held > 0:
            vorabpauschale_sum += tranche['original_amount'] * basiszins * 0.7 * (months_held / 12)
    actual_gains = balance - total_invested
    return min(vorabpauschale_sum, max(0.0, actual_gains))


class PrivateSavingPlan:
    def __init__(self, saving_contribution, horizon, frequency='monthly', initial_balance=0):
        """
        Initialize a Private Saving Plan
        
        Args:
            saving_contribution: Contribution amount (monthly if frequency='monthly', yearly if frequency='yearly')
            horizon: Investment horizon in years
            frequency: Investment frequency ('monthly' or 'yearly')
            initial_balance: Starting balance (default 0)
        """
        self.saving_contribution = saving_contribution
        self.horizon = horizon
        self.frequency = frequency
        self.initial_balance = initial_balance
        self.__average_yearly_growth = 0.07  # 7% default
        self.__ter = 0.005  # 0.5% default Total Expense Ratio
        self.__basiszins = 0.0229  # 2.29% default (2025 value)
        self.__tax_rate = 0.26375  # 26.375% (25% + 5.5% Soli)
        self.__freibetrag = 0  # €0 tax-free allowance
        self.__teilfreistellung = 0.7  # 70% for equity ETFs
    
    @property
    def average_yearly_growth(self):
        """Get the average yearly growth rate"""
        return self.__average_yearly_growth
    
    @average_yearly_growth.setter
    def average_yearly_growth(self, rate):
        """Set the average yearly growth rate"""
        self.__average_yearly_growth = rate
    
    @property
    def ter(self):
        """Get the Total Expense Ratio (TER)"""
        return self.__ter
    
    @ter.setter
    def ter(self, rate):
        """Set the Total Expense Ratio (TER)"""
        self.__ter = rate
    
    @property
    def basiszins(self):
        """Get the Basiszins for Vorabpauschale calculation"""
        return self.__basiszins
    
    @basiszins.setter
    def basiszins(self, rate):
        """Set the Basiszins for Vorabpauschale calculation"""
        self.__basiszins = rate
    
    @property
    def tax_rate(self):
        """Get the tax rate"""
        return self.__tax_rate
    
    @tax_rate.setter
    def tax_rate(self, rate):
        """Set the tax rate"""
        self.__tax_rate = rate
    
    @property
    def freibetrag(self):
        """Get the tax-free allowance (Freibetrag)"""
        return self.__freibetrag
    
    @freibetrag.setter
    def freibetrag(self, amount):
        """Set the tax-free allowance (Freibetrag)"""
        self.__freibetrag = amount

    @property
    def teilfreistellung(self):
        """Get the Teilfreistellung rate (taxable fraction: 0.7 for equity ETFs, 0.85 for mixed, 1.0 for bonds)"""
        return self.__teilfreistellung

    @teilfreistellung.setter
    def teilfreistellung(self, rate):
        """Set the Teilfreistellung rate"""
        self.__teilfreistellung = rate
    
    def calculate_balance(self, months=None, initial_balance=None):
        """
        Calculate balance after specified months with compound interest and Vorabpauschale
        
        Args:
            months: Number of months to calculate (default uses self.horizon * 12)
            initial_balance: Starting balance (default uses self.initial_balance)
        
        Returns:
            Dictionary with final balance, total invested, total Vorabpauschale paid, and yearly details
        """
        if months is None:
            months = self.horizon * 12
            
        if initial_balance is None:
            initial_balance = self.initial_balance
        
        if self.frequency == 'monthly':
            return self._calculate_monthly(months, initial_balance)
        elif self.frequency == 'yearly':
            return self._calculate_yearly(months, initial_balance)
        else:
            raise ValueError(f"Invalid frequency: {self.frequency}. Must be 'monthly' or 'yearly'")
    
    def _calculate_monthly(self, months, initial_balance=0):
        """Calculate balance with monthly contributions and compound interest"""
        # Net yearly growth after TER
        net_yearly_growth = self.__average_yearly_growth - self.__ter
        
        # Convert to monthly growth rate
        monthly_rate = (1 + net_yearly_growth) ** (1/12) - 1
        
        # Track each tranche (month purchased, original amount, current value)
        tranches = []
        if initial_balance > 0:
            tranches.append({
                'purchase_month': -1,
                'original_amount': initial_balance,
                'current_value': initial_balance
            })

        # Track yearly Vorabpauschale
        yearly_vorabpauschale = []
        cumulative_vorabpauschale_paid = 0
        cumulative_tax_paid = 0

        balance = initial_balance
        total_invested = initial_balance
        
        for month in range(months):
            # Apply growth to existing balance first (end-of-period contributions)
            for tranche in tranches:
                tranche['current_value'] *= (1 + monthly_rate)
            balance *= (1 + monthly_rate)
            
            # Then add new monthly contribution as a new tranche
            tranches.append({
                'purchase_month': month,
                'original_amount': self.saving_contribution,
                'current_value': self.saving_contribution
            })
            balance += self.saving_contribution
            total_invested += self.saving_contribution
            
            # Check if it's year-end (December, month 11, 23, 35, etc.)
            # Vorabpauschale is calculated on January 1st for the previous year
            if (month + 1) % 12 == 0:
                year = (month + 1) // 12
                vorabpauschale = self._calculate_yearly_vorabpauschale(tranches, month, balance, total_invested)
                
                # Teilfreistellung for equity ETFs: 30% exempt, 70% taxable
                taxable_after_teilfreistellung = vorabpauschale * self.__teilfreistellung
                taxable_amount = max(0, taxable_after_teilfreistellung - self.__freibetrag)
                tax_to_pay = taxable_amount * self.__tax_rate

                cumulative_vorabpauschale_paid += vorabpauschale
                cumulative_tax_paid += tax_to_pay

                yearly_vorabpauschale.append({
                    'year': year,
                    'vorabpauschale': vorabpauschale,
                    'tax_paid': tax_to_pay,
                    'balance_before_tax': balance
                })
        
        return {
            'final_balance': balance,
            'total_invested': total_invested,
            'total_vorabpauschale': cumulative_vorabpauschale_paid,
            'total_tax_paid': cumulative_tax_paid,
            'yearly_details': yearly_vorabpauschale
        }
    
    def _calculate_yearly(self, months, initial_balance=0):
        """Calculate balance with yearly contributions (Jan 1st) and yearly compound interest"""
        # Net yearly growth after TER
        net_yearly_growth = self.__average_yearly_growth - self.__ter
        
        # Track each tranche (month purchased, original amount, current value)
        tranches = []
        
        # Add initial balance as a tranche if present
        if initial_balance > 0:
            tranches.append({
                'purchase_month': -1,  # Before first period
                'original_amount': initial_balance,
                'current_value': initial_balance
            })
        
        # Track yearly Vorabpauschale
        yearly_vorabpauschale = []
        cumulative_vorabpauschale_paid = 0
        cumulative_tax_paid = 0
        
        balance = initial_balance
        total_invested = initial_balance
        
        years = months // 12
        
        for year in range(years):
            # Add yearly contribution on January 1st (month 0 of each year)
            yearly_amount = self.saving_contribution
            month = year * 12
            
            tranches.append({
                'purchase_month': month,
                'original_amount': yearly_amount,
                'current_value': yearly_amount
            })
            balance += yearly_amount
            total_invested += yearly_amount
            
            # Apply yearly growth to all tranches
            for tranche in tranches:
                tranche['current_value'] *= (1 + net_yearly_growth)
            balance *= (1 + net_yearly_growth)
            
            # Calculate Vorabpauschale at year-end
            vorabpauschale = self._calculate_yearly_vorabpauschale(tranches, (year + 1) * 12 - 1, balance, total_invested)
            
            # Teilfreistellung: taxable fraction (e.g. 0.7 for equity ETFs = 30% exempt)
            taxable_after_teilfreistellung = vorabpauschale * self.__teilfreistellung
            taxable_amount = max(0, taxable_after_teilfreistellung - self.__freibetrag)
            tax_to_pay = taxable_amount * self.__tax_rate

            cumulative_vorabpauschale_paid += vorabpauschale
            cumulative_tax_paid += tax_to_pay

            yearly_vorabpauschale.append({
                'year': year + 1,
                'vorabpauschale': vorabpauschale,
                'tax_paid': tax_to_pay,
                'balance_before_tax': balance
            })
        
        return {
            'final_balance': balance,
            'total_invested': total_invested,
            'total_vorabpauschale': cumulative_vorabpauschale_paid,
            'total_tax_paid': cumulative_tax_paid,
            'yearly_details': yearly_vorabpauschale
        }
    
    def _calculate_yearly_vorabpauschale(self, tranches, current_month, balance, total_invested):
        return calculate_yearly_vorabpauschale(
            tranches, current_month, balance, total_invested, self.__basiszins
        )

    def _calculate_months_held_in_year(self, purchase_month, current_month):
        return calculate_months_held_in_year(purchase_month, current_month)
        
# TO DO
# 1. unit test real calculation vorabpauschale with known values from internet examples
# 2. unit test compound interest calculation with known values from internet examples
