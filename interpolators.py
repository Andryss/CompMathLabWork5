import pandas as pd

from functions import *


class InterpolationResultEntity:
    name: str
    function: Function


class InterpolationResult:
    source_table: pd.DataFrame
    finite_differences_table: pd.DataFrame
    interpolation_entities: InterpolationResultEntity


def calculate_finite_differences(result: InterpolationResult):
    src_table = result.source_table
    n = src_table.shape[0]

    fd_table = pd.DataFrame({
        'x_i': src_table['x'],
        'y_i': src_table['y']
    })

    for i in range(n-1):
        index = f'delta^{i+1} y_i' if i != 0 else f'delta y_i'
        new_col = []
        last_col = fd_table[fd_table.columns[-1]]
        for j in range(n-1-i):
            new_col.append(last_col[j+1] - last_col[j])
        for k in range(1+i):
            new_col.append(pd.NA)
        fd_table[index] = new_col

    result.finite_differences_table = fd_table


class Interpolator:
    name: str

    def process(self, info: InterpolationResult) -> InterpolationResultEntity:
        raise Exception("Method is not overriden!")


class LagrangeInterpolator(Interpolator):
    name = "lagrange interpolator"

    def process(self, info: InterpolationResult) -> InterpolationResultEntity:
        result = InterpolationResultEntity()
        src_table = info.source_table
        src_table_x, src_table_y = src_table['x'], src_table['y']
        n = src_table.shape[0]

        def lagrange_at(x: float) -> float:
            res = 0.0
            for i in range(n):
                mul_i = src_table_y[i]
                for j in range(n):
                    if i == j:
                        continue
                    mul_i *= (x - src_table_x[j]) / (src_table_x[i] - src_table_x[j])
                res += mul_i
            return res

        result.name = "lagrange polynom"
        result.function = Function("???", lambda x: lagrange_at(x))
        return result


def get_all_interpolators() -> list[Interpolator]:
    return [
        LagrangeInterpolator()
    ]


def calculate_interpolations(result: InterpolationResult):
    interpolations = []
    for interpolator in get_all_interpolators():
        interpolations.append(interpolator.process(result))
    result.interpolation_entities = interpolations


def interpolate(table_func: TableFunction) -> InterpolationResult:
    result = InterpolationResult()
    result.source_table = table_func.table().copy()
    calculate_finite_differences(result)
    calculate_interpolations(result)
    return result


if __name__ == '__main__':
    table = TableFunction(pd.DataFrame({
        'x': [0.1, 0.2, 0.3, 0.4, 0.5],
        'y': [1.25, 2.38, 3.79, 5.44, 7.14]
    }))
    print(interpolate(table).finite_differences_table)
