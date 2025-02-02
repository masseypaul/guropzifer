# GuroPfizer

Development of a revolutionary solution to optimize Pfizer process and get a lot of money !!! :money_with_wings:

### Requirement

You will need numpy, pandas, gurobipy and scikitlearn

### Organization of the files

In the folder data, all the csv files used are stored.

For the step 1, you have:
- solve_pb.py (the basic resolution)
- solve_pb_epsilon.py (the epsilon constraint method)

For step 2, you have:
- solve_100.py (the solver for a bigger dataset)
- solve_pb_continuous.py (assign to multiple SR)
- new_SR (when adding new SR to the situation)

For step 3, you have:
- variable_offices.py (when offices for SR are not fixed)

For step 4, you have:
- UTA.py (as in the name of the file, the UTA method)