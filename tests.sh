#!/bin/bash

# Exclusive scan with a single thread block
echo "Test 1: Exclusive scan with a single thread block"
echo "Array of size 32, thread block of size 32"
python project-gpu.py input_test_32.txt --tb 32

echo "Test 2: Exclusive scan with a too large thread block"
echo "Array of size 32, thread block of size 64"
python project-gpu.py input_test_32.txt --tb 64

echo "Test 3: Exclusive scan with a non-power-of-2 array size and a single thread block"
echo "Array of size 30, thread block of size 32"
python project-gpu.py input_test_30.txt --tb 32

# Independent scans
echo "Test 4: Independent scans with multiple thread blocks"
echo "Array of size 64, thread block of size 8"
python project-gpu.py input_test_64.txt --tb 8 --independent

echo "Test 5: Independent scans with non-power-of-2 array size and multiple thread blocks"
echo "Array of size 62, thread block of size 8"
python project-gpu.py input_test_62.txt --tb 8 --independent

# Arbitrary scans
echo "Test 6: Arbitrary scans with multiple thread blocks"
echo "Array of size 64, thread block of size 8"
python project-gpu.py input_test_64.txt --tb 8

echo "Test 7: Arbitrary scans with non-power-of-2 array size and multiple thread blocks"
echo "Array of size 1023, thread block of size 8"
python project-gpu.py input_test_1023.txt --tb 8

# Array de taille 2^m avec un thread block de même taille (exemple : array de taille 32, thread block de taille 32)
echo "Array de taille 2^m avec un thread block de même taille"
python project-gpu.py input_test_32.txt --tb 32 --inclusive

# Array de taille 2^m avec un thread block trop grand (exemple : array de taille 32, thread block de taille 64)
echo "Array de taille 2^m avec un thread block trop grand"
python project-gpu.py input_test_32.txt --tb 64 --inclusive

# Array de taille !2^m avec un thread block unique (exemple : array de taille 30, thread block de taille 32)
echo "Array de taille !2^m avec un thread block unique"
python project-gpu.py input_test_30.txt --tb 32 --inclusive

# Scans indépendants de taille 2^m avec plusieurs thread blocks (exemple : array de taille 64, thread block de taille 8)
echo "Scans indépendants de taille 2^m avec plusieurs thread blocks"
python project-gpu.py input_test_64.txt --tb 8 --independent --inclusive

# Scans indépendants de taille !2^m avec plusieurs thread blocks (exemple : array de taille 62, thread block de taille 8)
echo "Scans indépendants de taille !2^m avec plusieurs thread blocks"
python project-gpu.py input_test_62.txt --tb 8 --independent --inclusive

# Scans arbitraires de taille 2^m avec plusieurs thread blocks (exemple : array de taille 64, thread block de taille 8)
echo "Scans arbitraires de taille 2^m avec plusieurs thread blocks"
python project-gpu.py input_test_64.txt --tb 8 --inclusive

# Scans arbitraires de taille !2^m avec plusieurs thread blocks (exemple : array de taille 1023, thread block de taille 8)
echo "Scans arbitraires de taille !2^m avec plusieurs thread blocks"
python project-gpu.py input_test_1023.txt --tb 8 --inclusive

# Très grands tableaux
echo "Très grands tableaux"
python project-gpu.py input_test_large.txt --inclusive

# Grands tableaux avec n'importe quelle taille de thread block
echo "Grands tableaux avec n'importe quelle taille de thread block"
python project-gpu.py input_test_large.txt --tb 64 --inclusive