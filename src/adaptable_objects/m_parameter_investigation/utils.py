from os import getcwd

def file_namer(n, r, d, version):
    path = getcwd()

    return (
        f"{path}/results/N"
        f"-{n if not isinstance(n, list) else '(' + str(min(n)) + '-' + str(max(n)) + ')x' + str(len(n))}"
        f"__R-{str(r).replace('.', '-') if not isinstance(r, list) else ('(' + str(min(r)) + '-' + str(max(r)) + ')x' + str(len(r))).replace('.', '-')}"
        f"__D-{d if not isinstance(d, list) else '(' + str(min(d)) + '-' + str(max(d)) + ')x' + str(len(d))}"
        f"__V-{version}.pkl"
    )

def temp_file_namer(n, experiment):
    path = getcwd()

    return (
        f"{path}/temp_save/N-{n}"
        f"__E-{experiment}.json"
    )
