import sys

import numpy


def usage():
    print("Usage: " + sys.argv[0] + " <power of 2>")


def random_array(power_of_two, minimum=-100, maximum=100):
    return numpy.random.randint(minimum, maximum, power_of_two, dtype=numpy.int32)


def generate_inputs(a):
    arr = random_array(a)
    print(arr)
    input_file = ("input_test_" + str(a) + ".txt")
    with open(input_file, "w") as f:
        t = numpy.array2string(arr, separator=",", threshold=arr.shape[0]).strip('[]').replace('\n', '').replace(
            ' ', '')
        f.write(t)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage()
        sys.exit(2)
    else:
        generate_inputs(int(sys.argv[1]))
