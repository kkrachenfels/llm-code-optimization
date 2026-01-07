#include <iostream>
#include <vector>
#include <chrono>
#include <random>

// Inefficient bubble sort implementation
void bubbleSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

int main() {
    const int SIZE = 10000;
    std::vector<int> data(SIZE);

    // Generate random data
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<> dis(1, 100000);

    for (int i = 0; i < SIZE; i++) {
        data[i] = dis(gen);
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    bubbleSort(data);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << duration.count() << std::endl;

    return 0;
}
