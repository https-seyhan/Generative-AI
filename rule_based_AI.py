def determine_action(value, thresholds):
    # Define a dictionary of rules using lambda functions
    rules = {
        'low': lambda x: 'Increase' if x < thresholds['low'] else None,
        'medium': lambda x: 'Maintain' if thresholds['low'] <= x < thresholds['high'] else None,
        'high': lambda x: 'Decrease' if x >= thresholds['high'] else None
    }
    
    # Apply the rules
    for rule in rules.values():
        action = rule(value)
        if action:
            return action
    return 'No action'

# Example thresholds
thresholds = {'low': 50, 'high': 80}

# Test the function
print(determine_action(30, thresholds))  # Output: Increase
print(determine_action(70, thresholds))  # Output: Maintain
print(determine_action(90, thresholds))  # Output: Decrease
