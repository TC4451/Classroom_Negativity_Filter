import sys

def print_to_stdout(*a):

    # Here a is the array holding the objects
    # passed as the argument of the function
    print(*a, file = sys.stdout)

# parameters for loading path
fn = ""
for file in sys.stdin:
    fn = file.split('/')[-1]
    fn = fn.split('.csv')[0]
    print_to_stdout(fn)
