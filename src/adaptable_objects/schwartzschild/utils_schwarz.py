from os import getcwd


def file_namer(n, d, version):
    path = getcwd()

    return (
        f"{path}/src/adaptable_objects/schwartzschild/results/N"
        f"-{n if not isinstance(n, list) else '(' + str(min(n)) + '-' + str(max(n)) + ')x' + str(len(n))}"
        f"__D-{d}"
        f"__V-{version}.pkl"
    )
