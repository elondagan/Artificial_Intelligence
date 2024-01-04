import pandas as pd

# load and clean data
results = pd.read_csv(r"results/1053elevator__lmc_custom_PLA.csv")
results = pd.DataFrame(results)
results = results.drop(columns=['problem I', 'problem G'])
results = results.dropna(how="all").reset_index(drop=True)
results_size = len(results)


names = ['lmc', 'custom', 'LA', 'PLA']


# computation time
for name in names:
    print(name + ":", sum(list(results[f'{name}_time'])))


# fill missing time values (with max time par problem)
for name in names:
    for i in range(len(results[f'{name}_time'])):
        if pd.isna(results[f'{name}_time'][i]):
            results[f'{name}_time'][i] = 120


# number of problems each algorithm solved within the time limit
problems_solved = []
time_limit = 300
for name in names:
    i = 0
    tot_time = 0
    solved = 0
    while True:
        cur_time = results[f'{name}_time'][i]
        tot_time += cur_time
        if tot_time >= time_limit:
            break
        if cur_time < 90:
            solved += 1
        i += 1
    problems_solved.append(solved)

print("problems solved:")
print(problems_solved)
print([300/i for i in problems_solved])

