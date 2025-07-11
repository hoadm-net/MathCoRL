"""
Enhanced Functions Module for Ablation Study

This module contains implementations of enhanced functions for ablation research.
These functions are completely separate from the original functions to ensure
proper isolation during experiments.

WARNING: This module is ONLY for ablation study. Do not import or use in
production code to maintain separation of concerns.
"""

import math
from typing import List, Union


# ==================== ORIGINAL BASIC MATH FUNCTIONS ====================
# Re-implement basic functions to ensure complete separation

def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return float(a) + float(b)


def sub(a: float, b: float) -> float:
    """Subtract second number from first number."""
    return float(a) - float(b)


def mul(a: float, b: float) -> float:
    """Multiply two numbers."""
    return float(a) * float(b)


def div(a: float, b: float) -> float:
    """Divide first number by second number."""
    return float(a) / float(b) if b != 0 else float('inf')


def pow(a: float, b: float) -> float:
    """Raise first number to the power of second number."""
    return float(a) ** float(b)


def sum(numbers: list) -> float:
    """Calculate the sum of a list of numbers."""
    import builtins
    return float(builtins.sum(float(x) for x in numbers))


def mean(numbers: list) -> float:
    """Calculate the arithmetic mean of a list of numbers."""
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def percentage(part: float, whole: float) -> float:
    """Calculate what percentage part is of whole."""
    if whole == 0:
        return 0.0
    return (float(part) / float(whole)) * 100.0


def round_num(a: float, digits: int = 0) -> float:
    """Round a number to given number of decimal places."""
    return round(float(a), digits)


def abs_num(a: float) -> float:
    """Get absolute value of a number."""
    return abs(float(a))


# Use different names to avoid conflicts with built-ins
builtin_sum = sum
builtin_round = round
builtin_abs = abs

# Override with our versions
round = round_num
abs = abs_num


# ==================== ENHANCED FINANCIAL FUNCTIONS ====================

def calculate_growth_rate(initial: float, final: float) -> float:
    """Calculate growth rate percentage.
    
    Args:
        initial: Initial value
        final: Final value
    
    Returns:
        Growth rate as percentage
    
    Example:
        calculate_growth_rate(100, 120) -> 20.0
    """
    if initial == 0:
        return 0.0
    return percentage(sub(final, initial), initial)


def calculate_interest(principal: float, rate: float, time: float) -> float:
    """Calculate simple interest.
    
    Args:
        principal: Principal amount
        rate: Interest rate (as percentage)
        time: Time period in years
    
    Returns:
        Interest amount
    
    Example:
        calculate_interest(1000, 5, 2) -> 100.0
    """
    rate_decimal = div(rate, 100.0)
    return mul(mul(principal, rate_decimal), time)


def calculate_compound_interest(principal: float, rate: float, time: float, compound_frequency: int = 1) -> float:
    """Calculate compound interest.
    
    Args:
        principal: Principal amount
        rate: Annual interest rate (as percentage)
        time: Time period in years
        compound_frequency: Compounding frequency per year
    
    Returns:
        Final amount after compound interest
    
    Example:
        calculate_compound_interest(1000, 5, 2, 1) -> 1102.5
    """
    rate_decimal = div(rate, 100.0)
    rate_per_period = div(rate_decimal, compound_frequency)
    total_periods = mul(compound_frequency, time)
    
    amount = mul(principal, pow(add(1, rate_per_period), total_periods))
    return amount


def calculate_present_value(future_value: float, rate: float, periods: int) -> float:
    """Calculate present value of future cash flow.
    
    Args:
        future_value: Future value
        rate: Discount rate (as percentage)
        periods: Number of periods
    
    Returns:
        Present value
    
    Example:
        calculate_present_value(1000, 10, 2) -> 826.45
    """
    rate_decimal = div(rate, 100.0)
    discount_factor = pow(add(1, rate_decimal), periods)
    return div(future_value, discount_factor)


def calculate_future_value(present_value: float, rate: float, periods: int) -> float:
    """Calculate future value of present cash flow.
    
    Args:
        present_value: Present value
        rate: Interest rate (as percentage)
        periods: Number of periods
    
    Returns:
        Future value
    
    Example:
        calculate_future_value(1000, 10, 2) -> 1210.0
    """
    rate_decimal = div(rate, 100.0)
    growth_factor = pow(add(1, rate_decimal), periods)
    return mul(present_value, growth_factor)


def calculate_annuity_payment(principal: float, rate: float, periods: int) -> float:
    """Calculate periodic payment for an annuity.
    
    Args:
        principal: Loan principal amount
        rate: Periodic interest rate (as percentage)
        periods: Number of payment periods
    
    Returns:
        Periodic payment amount
    
    Example:
        calculate_annuity_payment(10000, 5, 12) -> 1128.25
    """
    rate_decimal = div(rate, 100.0)
    if rate_decimal == 0:
        return div(principal, periods)
    
    numerator = mul(principal, rate_decimal)
    denominator = sub(1, pow(add(1, rate_decimal), -periods))
    return div(numerator, denominator)


def calculate_loan_balance(principal: float, rate: float, periods: int, payments_made: int) -> float:
    """Calculate remaining loan balance after some payments.
    
    Args:
        principal: Original loan amount
        rate: Periodic interest rate (as percentage)
        periods: Total number of periods
        payments_made: Number of payments already made
    
    Returns:
        Remaining loan balance
    
    Example:
        calculate_loan_balance(10000, 1, 60, 12) -> 8240.15
    """
    rate_decimal = div(rate, 100.0)
    if rate_decimal == 0:
        payment = div(principal, periods)
        return sub(principal, mul(payment, payments_made))
    
    payment = calculate_annuity_payment(principal, rate, periods)
    remaining_periods = sub(periods, payments_made)
    
    if remaining_periods <= 0:
        return 0.0
    
    return calculate_present_value(mul(payment, remaining_periods), rate, remaining_periods)


def calculate_depreciation_straight_line(cost: float, salvage_value: float, useful_life: int) -> float:
    """Calculate annual depreciation using straight-line method.
    
    Args:
        cost: Initial cost of asset
        salvage_value: Expected salvage value
        useful_life: Useful life in years
    
    Returns:
        Annual depreciation amount
    
    Example:
        calculate_depreciation_straight_line(10000, 1000, 5) -> 1800.0
    """
    if useful_life <= 0:
        return 0.0
    
    depreciable_amount = sub(cost, salvage_value)
    return div(depreciable_amount, useful_life)


def calculate_break_even_point(fixed_costs: float, price_per_unit: float, variable_cost_per_unit: float) -> float:
    """Calculate break-even point in units.
    
    Args:
        fixed_costs: Total fixed costs
        price_per_unit: Selling price per unit
        variable_cost_per_unit: Variable cost per unit
    
    Returns:
        Break-even point in units
    
    Example:
        calculate_break_even_point(10000, 50, 30) -> 500.0
    """
    contribution_margin = sub(price_per_unit, variable_cost_per_unit)
    if contribution_margin <= 0:
        return float('inf')
    
    return div(fixed_costs, contribution_margin)


def calculate_return_on_investment(gain: float, cost: float) -> float:
    """Calculate return on investment (ROI) percentage.
    
    Args:
        gain: Investment gain (profit)
        cost: Initial investment cost
    
    Returns:
        ROI as percentage
    
    Example:
        calculate_return_on_investment(2000, 10000) -> 20.0
    """
    if cost == 0:
        return 0.0
    
    return percentage(gain, cost)


def calculate_percentage(part: float, whole: float) -> float:
    """Calculate what percentage part is of whole (alias for percentage function).
    
    Args:
        part: Part value
        whole: Whole value
    
    Returns:
        Percentage value
    
    Example:
        calculate_percentage(765, 824) -> 92.8
    """
    return percentage(part, whole)


# ==================== TABLE PROCESSING FUNCTIONS ====================

def get_cell(table: list, row: int, col: int) -> float:
    """Get value from specific cell in table."""
    try:
        return float(table[row][col])
    except (IndexError, ValueError, TypeError):
        return 0.0


def get_column(table: list, col: int) -> list:
    """Get entire column from table."""
    try:
        return [float(row[col]) for row in table]
    except (IndexError, ValueError, TypeError):
        return []


def sum_column(table: list, col: int) -> float:
    """Calculate sum of values in a column."""
    column_values = get_column(table, col)
    return sum(column_values)


def average_column(table: list, col: int) -> float:
    """Calculate average of values in a column."""
    column_values = get_column(table, col)
    return mean(column_values)


def max_in_column(table: list, col: int) -> float:
    """Find maximum value in a column."""
    column_values = get_column(table, col)
    if not column_values:
        return 0.0
    return max(column_values)


def min_in_column(table: list, col: int) -> float:
    """Find minimum value in a column."""
    column_values = get_column(table, col)
    if not column_values:
        return 0.0
    return min(column_values)


def find_max_row(table: list, col: int) -> int:
    """Find row index with maximum value in specified column."""
    column_values = get_column(table, col)
    if not column_values:
        return -1
    
    max_value = max(column_values)
    for i, value in enumerate(column_values):
        if value == max_value:
            return i
    return -1


def find_min_row(table: list, col: int) -> int:
    """Find row index with minimum value in specified column."""
    column_values = get_column(table, col)
    if not column_values:
        return -1
    
    min_value = min(column_values)
    for i, value in enumerate(column_values):
        if value == min_value:
            return i
    return -1


def count_cells(table: list, col: int, condition_value: float) -> int:
    """Count cells in column that match a condition value."""
    column_values = get_column(table, col)
    return builtin_sum(1 for value in column_values if abs_num(value - condition_value) < 1e-6)


def table_lookup(table: list, search_col: int, search_value: float, return_col: int) -> float:
    """Look up value in table based on search criteria."""
    try:
        for row in table:
            if abs_num(float(row[search_col]) - search_value) < 1e-6:
                return float(row[return_col])
        return 0.0
    except (IndexError, ValueError, TypeError):
        return 0.0


# ==================== MISSING BASIC FUNCTIONS ====================

def min(*args: float) -> float:
    """Find the minimum value among given numbers."""
    import builtins
    return float(builtins.min(args))

def max(*args: float) -> float:
    """Find the maximum value among given numbers."""
    import builtins
    return float(builtins.max(args))

def floor(a: float) -> int:
    """Round down to nearest integer."""
    import math
    return math.floor(float(a))

def ceil(a: float) -> int:
    """Round up to nearest integer."""
    import math
    return math.ceil(float(a))

def median(numbers: list) -> float:
    """Calculate the median of a list of numbers."""
    if not numbers:
        return 0.0
    sorted_nums = sorted([float(x) for x in numbers])
    n = len(sorted_nums)
    if n % 2 == 0:
        return (sorted_nums[n//2 - 1] + sorted_nums[n//2]) / 2.0
    else:
        return sorted_nums[n//2]

def mode(numbers: list) -> float:
    """Find the most frequently occurring value."""
    if not numbers:
        return 0.0
    from collections import Counter
    counter = Counter([float(x) for x in numbers])
    return float(counter.most_common(1)[0][0])

def count(numbers: list) -> int:
    """Count the number of elements in a list."""
    return len(numbers)

def greater_than(a: float, b: float) -> bool:
    """Check if a is greater than b."""
    return float(a) > float(b)

def less_than(a: float, b: float) -> bool:
    """Check if a is less than b."""
    return float(a) < float(b)

def equal(a: float, b: float) -> bool:
    """Check if a equals b."""
    return abs(float(a) - float(b)) < 1e-9

def gcd(a: int, b: int) -> int:
    """Calculate greatest common divisor."""
    import math
    return math.gcd(int(a), int(b))

def lcm(a: int, b: int) -> int:
    """Calculate least common multiple."""
    import math
    return abs(int(a) * int(b)) // math.gcd(int(a), int(b))

def mod(a: float, b: float) -> float:
    """Get remainder of division."""
    return float(a) % float(b)

# ==================== MISSING FINANCIAL FUNCTIONS ====================

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values."""
    if old_value == 0:
        return 0.0
    return percentage(sub(new_value, old_value), old_value)

def calculate_compound_growth_rate(start_value: float, end_value: float, years: float) -> float:
    """Calculate compound annual growth rate (CAGR)."""
    if start_value == 0 or years == 0:
        return 0.0
    return sub(pow(div(end_value, start_value), div(1, years)), 1) * 100

def calculate_ratio(numerator: float, denominator: float) -> float:
    """Calculate ratio between two numbers."""
    if denominator == 0:
        return 0.0
    return div(numerator, denominator)

def calculate_net_change(current: float, previous: float) -> float:
    """Calculate net change between two values."""
    return sub(current, previous) 