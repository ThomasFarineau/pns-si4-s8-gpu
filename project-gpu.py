import sys
import numpy as np
from numba import cuda
import warnings
from numba.cuda.libdevice import log2f

warnings.filterwarnings("ignore")

"""
Auteur: Thomas FARINEAU - ft905846
Date: 2023-05-01
Description: Implémentation d'un scan exclusif sur GPU en utilisant CUDA.

Utilisation: python project-gpu.py <fichier d'entrée> [--tb <taille du bloc de threads>] [--independent] [--inclusive]
"""


def format_result(array):
    """
    Prend un tableau et retourne une chaîne de caractères correspondant au format de sortie souhaité.
    """
    return ', '.join(map(str, array))


def next_power_of_2(n):
    """
    Retourne la prochaine puissance de 2 supérieure ou égale à n.
    """
    return 1 << (n - 1).bit_length()


def process_args():
    """
    Traite les arguments de la ligne de commande et renvoie les options sous forme de dictionnaire.
    """
    input_file = sys.argv[1]
    options = {"input_file": input_file, "thread_block": 0, "independent": False, "inclusive": False}

    args_iter = iter(sys.argv[2:])
    for arg in args_iter:
        if arg == "--tb":
            options["thread_block"] = int(next(args_iter))
        elif arg == "--independent":
            options["independent"] = True
        elif arg == "--inclusive":
            options["inclusive"] = True

    return options


def read_input(input_file):
    """
    Lit le fichier d’entrée et retourne le tableau sous forme de liste.
    """
    with open(input_file) as file:
        return eval(next(file))


def scan_gpu(array, thread_block, independent, inclusive):
    """
    Exécute un scan exclusif sur le tableau d’entrée en utilisant les options spécifiées.
    """
    n = len(array)
    if n <= 1:
        return [0]

    if thread_block == 0:
        thread_block = n

    block_size = np.ceil(n / thread_block).astype(np.int32)
    d_array = cuda.to_device(np.array(array, dtype=np.int32))
    d_intermediate_array = cuda.to_device(np.zeros(block_size, dtype=np.int32))

    scan_kernel[block_size, thread_block](d_array, n, d_intermediate_array)
    cuda.synchronize()

    array = d_array.copy_to_host()
    intermediate_array = d_intermediate_array.copy_to_host()

    if not independent:
        intermediate_array = scan_gpu(intermediate_array, thread_block, False, inclusive)
        d_array = cuda.to_device(np.array(array, dtype=np.int32))
        d_intermediate_array = cuda.to_device(np.array(intermediate_array, dtype=np.int32))

        add_sums_kernel[block_size, thread_block](d_array, d_intermediate_array, n, thread_block)
        cuda.synchronize()

        array = d_array.copy_to_host()

    if inclusive:
        array = inclusive_scan(array, thread_block, independent, intermediate_array)

    return array


def inclusive_scan(array, thread_block, independent, intermediate_array):
    """
    Transforme le résultat d’un scan exclusif en un scan inclusif.
    """
    for i in range(len(array) - 1):
        array[i] = array[i + 1]

        if (i + 1) % thread_block == 0 and independent:
            array[i] = intermediate_array[i // thread_block]

    array[-1] = intermediate_array[-1]
    return array


@cuda.jit(device=True)
def device_next_power_of_2(n):
    if n and not (n & (n - 1)):
        return n

    count = 0
    while n != 0:
        n >>= 1
        count += 1

    return 1 << count


@cuda.jit
def scan_kernel(array, n, intermediate_array):
    thread_id = cuda.threadIdx.x
    global_id = thread_id + cuda.blockIdx.x * cuda.blockDim.x
    shared_array = cuda.shared.array(1023, dtype=np.int32)
    # Copier la mémoire globale dans la mémoire partagée
    if thread_id < n:
        shared_array[thread_id] = array[global_id]

    if thread_id + n < device_next_power_of_2(n):
        shared_array[thread_id + n] = 0

    n = device_next_power_of_2(n)
    m = log2f(n)
    cuda.syncthreads()

    # Montée
    for d in range(m):
        k = thread_id * int(2 ** (d + 1))
        if k < n - 1:
            shared_array[k + int(2 ** (d + 1)) - 1] += shared_array[k + int(2 ** d) - 1]
        cuda.syncthreads()

    # Dernier élément
    if thread_id == 0:
        intermediate_array[cuda.blockIdx.x] = shared_array[n - 1]
        shared_array[n - 1] = 0
    cuda.syncthreads()

    # Descente
    for d in range(m - 1, -1, -1):
        k = thread_id * int(2 ** (d + 1))
        if k < n - 1:
            t = shared_array[k + int(2 ** d) - 1]
            shared_array[k + int(2 ** d) - 1] = shared_array[k + int(2 ** (d + 1)) - 1]
            shared_array[k + int(2 ** (d + 1)) - 1] += t
        cuda.syncthreads()

    # Copier la mémoire partagée dans la mémoire globale
    if thread_id < n:
        array[global_id] = shared_array[thread_id]
    cuda.syncthreads()


@cuda.jit
def add_sums_kernel(sums_array, intermediate_array, n, thread_block_size):
    global_id = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if global_id < n:
        sums_array[global_id] += intermediate_array[global_id // thread_block_size]


if __name__ == '__main__':
    options = process_args()
    input_file = options["input_file"]
    thread_block = options["thread_block"]
    independent = options["independent"]
    inclusive = options["inclusive"]
    array = read_input(input_file)
    result = scan_gpu(array, thread_block, independent, inclusive)
    print(format_result(result))
