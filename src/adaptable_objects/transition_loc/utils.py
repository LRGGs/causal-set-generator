from os import getcwd


def file_namer(n, d, version):
    path = getcwd()
    print(path)

    return (
        f"{path}/results/N"
        f"-{n if not isinstance(n, list) else '(' + str(min(n)) + '-' + str(max(n)) + ')x' + str(len(n))}"
        f"__D-{d}"
        f"__V-{version}.pkl"
    )


print(file_namer(10, 2, 1))
