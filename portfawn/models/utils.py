import logging

import numpy as np

logger = logging.getLogger(__name__)

DAYS_IN_YEAR = 252


def annual_to_daily_return(annual_return, compounding=True):
    """
    Convert annual return to daily return.

    Args:
        annual_return (float): Annual return (as a decimal, e.g., 0.1 for 10%).
        compounding (bool): Whether to account for compounding. Defaults to True.

    Returns:
        float: Daily return (as a decimal).
    """
    if compounding:
        return (1 + annual_return) ** (1 / DAYS_IN_YEAR) - 1
    else:
        return annual_return / DAYS_IN_YEAR


def annual_to_daily_volatility(annual_volatility):
    """
    Convert annual standard deviation (volatility) to daily volatility.

    Args:
        annual_volatility (float): Annual volatility (as a decimal, e.g., 0.2 for 20%).

    Returns:
        float: Daily volatility (as a decimal).
    """
    return annual_volatility / np.sqrt(DAYS_IN_YEAR)


def validate_and_update_target_sd(target_sd, quadratic_biases):
    """
    Validate and update the target standard deviation (_target_sd) if necessary.

    Args:
        target_sd (float): The target standard deviation to validate.
        quadratic_biases (np.array): The covariance matrix of asset returns.

    Returns:
        float: Updated (or original) target standard deviation that is within the feasible range.
    """
    # Compute minimum and maximum variances (assuming boundary weights)
    min_variance = np.min(
        np.diag(quadratic_biases)
    )  # Smallest variance for single asset
    max_variance = np.max(np.sum(quadratic_biases, axis=1))  # Maximum possible variance
    min_sd = np.sqrt(min_variance)
    max_sd = np.sqrt(max_variance)

    # Update target_sd if it's out of range
    if target_sd < min_sd:
        updated_sd = min_sd
        logger.info(
            f"Target standard deviation {target_sd:.4f} is too low. "
            f"Updating to minimum feasible value {updated_sd:.4f}."
        )
    elif target_sd > max_sd:
        updated_sd = max_sd
        logger.info(
            f"Target standard deviation {target_sd:.4f} is too high. "
            f"Updating to maximum feasible value {updated_sd:.4f}."
        )
    else:
        updated_sd = target_sd  # Valid target_sd
        logger.info(f"Target standard deviation {target_sd:.4f} is valid.")

    return updated_sd


def validate_and_update_target_return(target_return, linear_biases):
    """
    Validate and update the target return (_target_return) if necessary.

    Args:
        target_return (float): The target return to validate.
        linear_biases (np.array): Expected returns for the assets.

    Returns:
        float: Updated (or original) target return that is within the feasible range.
    """
    # Compute the minimum and maximum achievable returns
    min_return = np.min(linear_biases)  # Return from the least performing asset
    max_return = np.max(linear_biases)  # Return from the best performing asset

    # Update target_return if it's out of range
    if target_return < min_return:
        updated_return = min_return
        logger.info(
            f"Target return {target_return:.4f} is too low. "
            f"Updating to minimum feasible value {updated_return:.4f}."
        )
    elif target_return > max_return:
        updated_return = max_return
        logger.info(
            f"Target return {target_return:.4f} is too high. "
            f"Updating to maximum feasible value {updated_return:.4f}."
        )
    else:
        updated_return = target_return  # Valid target_return
        logger.info(f"Target return {target_return:.4f} is valid.")

    return updated_return
