#include <sstream>
#include <fstream>
#include <cassert>
#include <string>
#include <iostream>
#include <stdexcept>

#define ASSERT(cond, msg, ...) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "[%s: %d line] " msg "\n\n", __FILE__, __LINE__, ##__VA_ARGS__); \
            assert(cond); \
        } \
    } while (0)

// device strict
struct Layer {
    float *w;
    float *b;
    float *dz;
    float *x;
    float *a;
} layer[4];

constexpr int _ceil(const int a, const int b) {
    return (a + b - 1) / b;
}

inline int argmax(const float *__restrict__ arr, const int len) {
    int max_idx = 0;
    if (len <= 0) return -1;
    float max_val = arr[0];
    for (int i = 1; i < len; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    return max_idx;
}

inline void read_dataset(const char *path, 
    const int start, 
    const int length, 
    float *__restrict__ X, 
    float *__restrict__ y
) {
    constexpr int input_size = 784;
    constexpr int labels = 10;
    try {
        std::ifstream fin(path);
        if (!fin.is_open()) {
            throw std::runtime_error("Failed to open \"" + std::string(path) + "\"");
        }
        std::string line;
        int line_cur = 0;
        int sample_idx = 0;
        // Immutable lambda binding: 'line_loc_msg' cannot be reassigned,
        // but it captures 'line_cur' by reference, allowing dynamic evaluation.
        const auto line_loc_msg = [&] { return " at line " + std::to_string(line_cur); };
        while (sample_idx < length && std::getline(fin, line)) {
            if (line_cur++ < start) continue;
            const char *ptr = line.c_str();
            char *end;
            
            // 1. Parse label
            int label = std::strtol(ptr, &end, labels);
            if (ptr == end) {
                throw std::runtime_error("Failed to parse label" + line_loc_msg());
            }
            if (label < 0 || label >= labels) {
                throw std::runtime_error("Label out of range" + line_loc_msg());
            }
            ptr = end;
            if (*ptr == ',') {
                ptr++;
            } else {
                throw std::runtime_error("Expected comma after label" + line_loc_msg());
            }
            
            // 2. Write one-hot y
            float *y_row = y + sample_idx * labels;
            std::memset(y_row, 0, labels * sizeof(float));
            y_row[label] = 1.0f;
            
            // 3. Parse x_row
            float *X_row = X + sample_idx * input_size;
            for (int j = 0; j < input_size; j++) {
                while (*ptr && (*ptr == ' ' || *ptr == '\t')) ptr++;
                float val = std::strtof(ptr, &end);
                if (ptr == end) {
                    throw std::runtime_error("Failed to parse pixel " + std::to_string(j) + line_loc_msg());
                }
                ptr = end;
                X_row[j] = val / 255.0f;
                if (j < input_size - 1) {
                    if (*ptr != ',') {
                        throw std::runtime_error("Expected ',' after pixel " + std::to_string(j) + line_loc_msg());
                    }
                    ptr++;
                }
            }
            ++sample_idx;
            while (*ptr && (*ptr == ' ' || *ptr == '\t' || *ptr == '\r')) ptr++;
            if (*ptr != '\0') {
                throw std::runtime_error("Unexpected data after last pixel" + line_loc_msg());
            }
        }
        if (sample_idx < length) {
            throw std::runtime_error("Not enough data rows. Expected: " + std::to_string(length) + ", got: " + std::to_string(sample_idx));
        }
    } catch (const std::exception &e) {
        std::cerr << "read_dataset()" << " Error: " << e.what() << '\n';
        std::exit(EXIT_FAILURE);
    }
}