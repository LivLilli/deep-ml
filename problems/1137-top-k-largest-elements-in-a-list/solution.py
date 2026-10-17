def top_three_largest(values):
    # values: list of numbers
    # return the three largest values in descending order
    
    if len(values) ==0:
        return []
        
    if len(values) ==1:
        return values
        
    first = float('-inf')
    second = float('-inf')
    third = float('-inf')
    
    for el in values:
        if el > first: 
            third = second
            second = first
            first = el
        elif el > second:
            third = second
            second = el
        elif el > third: 
            third = el
    if len(values) ==2:
        return [first, second]
    return [first, second, third]