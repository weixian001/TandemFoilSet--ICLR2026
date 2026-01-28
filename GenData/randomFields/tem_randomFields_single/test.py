#import math
from scipy.optimize import fsolve
def compute_parameters_from_start_end_total_length(start_width, end_width, total_length):
    """
    Calculate the missing parameters based on total length, start cell width and end cell width.
    Returns:
    - Number of Cells
    - Total Expansion Ratio
    - Cell-to-Cell Expansion Ratio
    """
    # Calculate total expansion ratio
    total_expansion_ratio = end_width / start_width

    # Define function to solve for N using fsolve
    def equation(x): # [r,N]
        return [ end_width - start_width*x[0]**(x[1]-1),
                total_length - ( start_width * (1 - x[0]**x[1]) / (1 - x[0]) ) ]

    # Solve for N
    r, N = fsolve(equation, [1.1,100])#[0]  # starting with an initial guess of 10

    # Calculate cell-to-cell expansion ratio using the formula derived above
    #r = (end_width/start_width)**(1/(N-1))

    # Return calculated values
    return round(N), total_expansion_ratio, r


def compute_parameters_from_cells_start_end(N, start_width, end_width):
    total_expansion_ratio = end_width / start_width
    r = (end_width / start_width) ** (1 / (N - 1))
    total_length = start_width * (1 - r**N) / (1 - r)
    return total_length, total_expansion_ratio, r

def compute_parameters_from_cells_start_total_length(N, start_width, total_length):
    def equation(r):
        return start_width * (1 - r**N) / (1 - r) - total_length
    r = fsolve(equation, 1.1)[0]
    end_width = start_width * r**(N-1)
    total_expansion_ratio = end_width / start_width
    return end_width, total_expansion_ratio, r
def compute_parameters_from_cells_total_end(N, total_length, end_width):
    def equation(r):
        delta_1 = end_width / (r ** (N-1))
        return delta_1 * (1 - r**N) / (1 - r) - total_length
    r = fsolve(equation, 1.1)[0]
    start_width = end_width / r**(N-1)
    total_expansion_ratio = end_width / start_width
    return start_width, total_expansion_ratio, r

def compute_parameters_from_cells_total_start(N, total_length, start_width):
    def equation(r):
        return start_width * (1 - r**N) / (1 - r) - total_length
    r = fsolve(equation, 1.1)[0]
    end_width = start_width * r**(N-1)
    total_expansion_ratio = end_width / start_width
    return end_width, total_expansion_ratio, r

def compute_parameters_from_start_end_ratio(start_width, end_width, total_expansion_ratio):
    r = total_expansion_ratio ** (1 / (end_width / start_width - 1))
    N = fsolve(lambda n: start_width * (1 - r**n) / (1 - r) - (end_width - start_width) / (r - 1), 10)[0]
    total_length = start_width * (1 - r**N) / (1 - r)
    return int(round(N)), total_length, r

# ... Similarly, define functions for the other three combinations

def blockMesh_calculator(N=None, total_length=None, start_width=None, end_width=None, total_expansion_ratio=None, cell_to_cell_expansion=None):
    # Check for the first combination: start_width, end_width, total_length
    if start_width is not None and end_width is not None and total_length is not None:
        return compute_parameters_from_start_end_total_length(start_width, end_width, total_length)

    # Check for the second combination: N, start_width, end_width
    elif N is not None and start_width is not None and end_width is not None:
        return compute_parameters_from_cells_start_end(N, start_width, end_width)

    # Check for the third combination: N, start_width, total_length
    elif N is not None and start_width is not None and total_length is not None:
        return compute_parameters_from_cells_start_total_length(N, start_width, total_length)

    # Check for the fourth combination: N, total_length, end_width
    elif N is not None and total_length is not None and end_width is not None:
        return compute_parameters_from_cells_total_end(N, total_length, end_width)

    # Check for the fifth combination: N, total_length, start_width
    elif N is not None and total_length is not None and start_width is not None:
        return compute_parameters_from_cells_total_start(N, total_length, start_width)

    # Check for the sixth combination: start_width, end_width, total_expansion_ratio
    elif start_width is not None and end_width is not None and total_expansion_ratio is not None:
        return compute_parameters_from_start_end_ratio(start_width, end_width, total_expansion_ratio)

    else:
        raise ValueError("Invalid combination of parameters provided.")
# ... (other import statements and function definitions)

if __name__ == "__main__":
    import sys

    # Create a dictionary to map arguments
    params = {
        "N": None,
        "total_length": None,
        "start_width": None,
        "end_width": None,
        "total_expansion_ratio": None,
        "cell_to_cell_expansion": None
    }

    for arg in sys.argv[1:]:
        key, value = arg.split("=")
        if key in params:
            if '.' in value:
                params[key] = float(value)
            elif value.isdigit():
                params[key] = int(value)
            else:
                params[key] = None

    result1, result2, result3 = blockMesh_calculator(**params)
    print(result1, result2, result3)

