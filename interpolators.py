import math

from functions import *


class InterpolationResultEntity:
    name: str


class InterpolationResultEntityError(InterpolationResultEntity):
    error: Exception


class InterpolationResultEntitySuccess(InterpolationResultEntity):
    function: Function


class InterpolationResult:
    source_table: pd.DataFrame
    finite_differences_table: pd.DataFrame
    interpolation_entities: list[InterpolationResultEntity]


def calculate_finite_differences(result: InterpolationResult):
    src_table = result.source_table.sort_values('x').reset_index(drop=True)
    n = src_table.shape[0]

    fd_table = pd.DataFrame({
        'x_i': src_table['x'],
        'y_i': src_table['y']
    })

    for i in range(n-1):
        index = f'delta^{i+1} y_i'
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
        src_table = info.source_table.sort_values('x').reset_index(drop=True)
        src_table_x, src_table_y = src_table['x'], src_table['y']
        n = src_table.shape[0] - 1

        def lagrange_at(x: float) -> float:
            res = 0.0
            for i in range(n+1):
                mul_i = src_table_y[i]
                for j in range(n+1):
                    if i == j:
                        continue
                    mul_i *= (x - src_table_x[j]) / (src_table_x[i] - src_table_x[j])
                res += mul_i
            return res

        result = InterpolationResultEntitySuccess()
        result.name = "lagrange polynom"
        result.function = Function("???", lambda x: lagrange_at(x))
        return result


class NewtonInterpolatorWithEqualDistance(Interpolator):
    name = "newton interpolator with equal distance"

    def process(self, info: InterpolationResult) -> InterpolationResultEntity:
        src_table = info.source_table.sort_values('x').reset_index(drop=True)
        finite_diffs_table = info.finite_differences_table
        src_table_x, src_table_y = src_table['x'], src_table['y']
        n = src_table.shape[0] - 1

        h = src_table_x[1] - src_table_x[0]

        threshold = src_table_x.min() + (src_table_x.max() - src_table_x.min()) / 2

        def newton_equal_dist_at(x: float) -> float:
            if x <= threshold:
                return newton_left_at((x - src_table_x[0]) / h)
            else:
                return newton_right_at((x - src_table_x[n]) / h)

        def newton_left_at(t: float) -> float:
            res = src_table_y[0]
            for i in range(1, n + 1):
                add = finite_diffs_table[f'delta^{i} y_i'][0]
                for j in range(i):
                    add *= (t - j)
                add /= math.factorial(i)
                res += add
            return res

        def newton_right_at(t: float) -> float:
            res = src_table_y[n]
            for i in range(1, n + 1):
                add = finite_diffs_table[f'delta^{i} y_i'][n - i]
                for j in range(i):
                    add *= (t + j)
                add /= math.factorial(i)
                res += add
            return res

        result = InterpolationResultEntitySuccess()
        result.name = "newton polynom (eq dist)"
        result.function = Function("???", lambda x: newton_equal_dist_at(x))
        return result


class NewtonInterpolatorWithNonEqualDistance(Interpolator):
    name = "newton interpolator with non equal distance"

    def process(self, info: InterpolationResult) -> InterpolationResultEntity:
        src_table = info.source_table.sort_values('x').reset_index(drop=True)
        src_table_x, src_table_y = src_table['x'], src_table['y']
        n = src_table.shape[0] - 1

        def newton_non_equal_dist_at(x: float) -> float:
            res = src_table_x[0]
            for k in range(1, n + 1):
                add = divided_difference(0, k)
                for j in range(k):
                    add *= (x - src_table_x[j])
                res += add
            return res

        def divided_difference(i: int, k: int) -> float:
            if k == 0:
                return src_table_y[i]
            return (divided_difference(i + 1, k - 1) - divided_difference(i, k - 1)) / (
                        src_table_x[i + k] - src_table_x[i])

        result = InterpolationResultEntitySuccess()
        result.name = "newton polynom (non eq dist)"
        result.function = Function("???", lambda x: newton_non_equal_dist_at(x))
        return result


def is_equal_dist(table_x: pd.Series, threshold: float = 1e-5) -> bool:
    vals = table_x.values
    min_dist = max_dist = vals[1] - vals[0]
    for i in range(1, table_x.size - 1):
        dist = vals[i+1] - vals[i]
        min_dist = min(min_dist, dist)
        max_dist = max(max_dist, dist)
    return (max_dist - min_dist) < threshold


class NewtonInterpolator(Interpolator):
    name = "newton interpolator"

    equal_dist_interpolator: Interpolator = NewtonInterpolatorWithEqualDistance()
    non_equal_dist_interpolator: Interpolator = NewtonInterpolatorWithNonEqualDistance()

    def process(self, info: InterpolationResult) -> InterpolationResultEntity:
        src_table = info.source_table.sort_values('x').reset_index(drop=True)
        equal_dist = is_equal_dist(src_table['x'])

        if equal_dist:
            return self.equal_dist_interpolator.process(info)
        else:
            return self.non_equal_dist_interpolator.process(info)


class GaussInterpolator(Interpolator):
    name = "gauss interpolator"

    def process(self, info: InterpolationResult) -> InterpolationResultEntity:
        src_table = info.source_table.sort_values('x').reset_index(drop=True)
        finite_diffs_table = info.finite_differences_table
        src_table_x, src_table_y = src_table['x'], src_table['y']

        if not is_equal_dist(src_table_x):
            result = InterpolationResultEntityError()
            result.name = "gauss polynom"
            result.error = Exception("Can't use gauss interpolator with not equal intervals")
            return result

        i_center = (src_table.shape[0] - 1) // 2
        n_negative = i_center

        a = src_table_x[i_center]
        h = src_table_x[1] - src_table_x[0]

        def gauss_at(x: float) -> float:
            if x < a:
                return gauss_left_half_at((x - a) / h)
            elif x > a:
                return gauss_right_half_at((x - a) / h)
            else:
                return src_table_y[i_center]

        def gauss_left_half_at(t: float) -> float:
            res = src_table_y[i_center]
            for i in range(1, 2 * n_negative + 1):
                index = (i + 1) // 2
                add = finite_diffs_table[f'delta^{i} y_i'][i_center - index]
                for j in range(-index + 1, (i // 2) + 1):
                    add *= (t + j)
                add /= math.factorial(i)
                res += add
            return res

        def gauss_right_half_at(t: float) -> float:
            res = src_table_y[i_center]
            for i in range(1, 2*n_negative+2):
                index = i // 2
                add = finite_diffs_table[f'delta^{i} y_i'][i_center - index]
                for j in range(-index, (i+1) // 2):
                    add *= (t + j)
                add /= math.factorial(i)
                res += add
            return res

        result = InterpolationResultEntitySuccess()
        result.name = "gauss polynom"
        result.function = Function("???", lambda x: gauss_at(x))
        return result


def get_all_interpolators() -> list[Interpolator]:
    return [
        LagrangeInterpolator(),
        NewtonInterpolator(),
        GaussInterpolator()
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
