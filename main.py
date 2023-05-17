import sys
import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from interpolators import *


def read_int_from_console(number_name: str) -> int:
    print(f"\nEnter {number_name}:")
    line = input().strip()
    try:
        return int(line)
    except Exception as e:
        raise Exception(f"Can't read int value: {e.__str__()}")


def read_float_from_console(number_name: str) -> float:
    print(f"\nEnter {number_name}:")
    line = input().strip()
    try:
        return float(line)
    except Exception as e:
        raise Exception(f"Can't read float value: {e.__str__()}")


def read_point_from_console() -> [float, float]:
    try:
        args = input().strip().split()
        assert len(args) == 2, "point must have 2 float coordinates"
        return [float(args[0]), float(args[1])]
    except Exception as e:
        raise Exception(f"Can't read point from console: {e.__str__()}")


def read_table_function_from_console() -> TableFunction:
    points_count = read_int_from_console("points count")
    print("\nEnter points coordinates sequentially (in each row like \"<x> <y>\"):")
    points = []
    for i in range(points_count):
        points.append(read_point_from_console())
    try:
        return TableFunction(pd.DataFrame(data=np.array(points), columns=['x', 'y']))
    except Exception as e:
        raise Exception(f"Can't read table function from console: {e.__str__()}")


def read_table_function_from_file() -> TableFunction:
    print("\nEnter the filename you want to read from:")
    filename = input().strip()
    try:
        values = pd.read_csv(filename, header=None).values
        data_table = pd.DataFrame(data=values, columns=['x', 'y'])
        assert len(data_table) >= 5, "Must be at least 5 points"
        return TableFunction(data_table)
    except Exception as e:
        raise Exception("file \"" + filename + "\" can't be opened: " + e.__str__())


def get_all_existing_functions() -> list[Function]:
    return [
        Function('sin(x)', lambda x: math.sin(x))
    ]


def read_interval() -> [float, float]:
    line = input("\nEnter the interval boundaries:\n").strip()
    interval: [float, float]
    try:
        interval = [float(x) for x in line.split()]
        if len(interval) != 2 or interval[1] < interval[0]:
            raise Exception("not an interval")
        return interval
    except Exception as e:
        raise Exception(f"can't read interval: {e.__str__()}")


def read_table_function_from_existing_function():
    print("\nChoose the function:")
    functions = get_all_existing_functions()
    for i, function in enumerate(functions):
        print(f'{i}\t{function.__str__()}')
    line = input("(enter the number) ").strip()
    try:
        function = functions[int(line)]
        [left, right] = read_interval()
        points_count = read_int_from_console("number of points on the interval")

        x_coords = np.linspace(start=left, stop=right, num=points_count)
        y_mapping = np.vectorize(lambda x: function.at(x))
        y_coords = y_mapping(x_coords)

        return TableFunction(pd.DataFrame({
            'x': x_coords,
            'y': y_coords
        }))
    except Exception as e:
        raise Exception(f"can't get table from given function: {e.__str__()}")


def read_table_function() -> TableFunction:
    print("\nChoose the method you want to read the table:")
    print("0\tread from console")
    print("1\tread from file")
    print("2\tread from existing function")
    line = input("(enter the number) ").strip()
    if line == "0":
        return read_table_function_from_console()
    elif line == "1":
        return read_table_function_from_file()
    elif line == "2":
        return read_table_function_from_existing_function()
    else:
        raise Exception("No such option :(")


def print_source_table(interpolation_result: InterpolationResult):
    print(f"\nSource function is: \n{interpolation_result.source_table.T}")


def print_finite_differences_table(interpolation_result: InterpolationResult):
    print(f'\nFinite differences is: \n{interpolation_result.finite_differences_table}')


def show_interpolation_plot(interpolation_result: InterpolationResult, extrapolation_point: float):
    warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

    x_src, y_src = interpolation_result.source_table['x'], interpolation_result.source_table['y']

    x_left, x_right = min(x_src.min(), extrapolation_point), max(x_src.max(), extrapolation_point)
    for interpolation_entity in interpolation_result.interpolation_entities:
        if isinstance(interpolation_entity, InterpolationResultEntitySuccess):
            fig, ax = plt.subplots()

            function = interpolation_entity.function

            x_coords = np.linspace(start=x_left, stop=x_right, num=2000)
            y_mapping = np.vectorize(lambda x: function.at(x))
            y_coords = y_mapping(x_coords)

            ax.scatter(x_coords, y_coords, label=interpolation_entity.name)
            plt.scatter(x_src, y_src, c='red', label='source points')

            extrapolation_result = function.at(extrapolation_point)
            print(f"\nWhen calculating {interpolation_entity.name}:")
            print(f"extrapolation result for {extrapolation_point} is {extrapolation_result}")
            plt.scatter(extrapolation_point, extrapolation_result, c='yellow', label='extrapolation')

            ax.set_title(interpolation_entity.name)
            ax.legend()
        elif isinstance(interpolation_entity, InterpolationResultEntityError):
            error = interpolation_entity.error
            print(f"\nSome error happened when calculating {interpolation_entity.name}:")
            print(f"{error}")

    plt.show()


def print_interpolation_result(interpolation_result: InterpolationResult, extrapolation_point: float):
    if interpolation_result.source_table.shape[0] < 30:
        pd.set_option('display.expand_frame_repr', False)

    print("\nHere is interpolation result:")
    print_source_table(interpolation_result)
    print_finite_differences_table(interpolation_result)
    show_interpolation_plot(interpolation_result, extrapolation_point)


def read_extrapolation_point() -> float:
    return read_float_from_console("point you want to extrapolate")


def run():
    try:
        table_function = read_table_function()

        if table_function.table().shape[0] > 20:
            raise Exception("Too much points (unstable behavior, result not guaranteed)")

        interpolation_result = interpolate(table_function)
        extrapolation_point = read_extrapolation_point()
        print_interpolation_result(interpolation_result, extrapolation_point)
    except Exception as e:
        print(f'Some error occurred: {e}', file=sys.stderr)


if __name__ == '__main__':
    run()
