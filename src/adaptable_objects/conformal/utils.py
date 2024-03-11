from os import getcwd

def file_namer(n, d, i):
    path = getcwd()

    return (
        f"{path}/results/N"
        f"-{n if not isinstance(n, list) else '(' + str(min(n)) + '-' + str(max(n)) + ')x' + str(len(n))}"
        f"__D-{d if not isinstance(d, list) else '(' + str(min(d)) + '-' + str(max(d)) + ')x' + str(len(d))}"
        f"__I-{i}.pkl"
    )
