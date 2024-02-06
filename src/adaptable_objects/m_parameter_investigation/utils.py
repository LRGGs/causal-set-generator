from os import getcwd

def file_namer(n, r, d, version):
    path = getcwd().split("src")[0]

    return (
        f"{path}/adaptable_objects/m_parameter_investigation/results/N"
        f"-{n if not isinstance(n, list) else '(' + str(min(n)) + '-' + str(max(n)) + ')x' + str(len(n))}"
        f"__R-{str(r).replace('.', '-') if not isinstance(r, list) else ('(' + str(min(r)) + '-' + str(max(r)) + ')x' + str(len(r))).replace('.', '-')}"
        f"__D-{d if not isinstance(d, list) else '(' + str(min(d)) + '-' + str(max(d)) + ')x' + str(len(d))}"
        f"__V-{version}.pkl"
    )
