NVCC = nvcc
TARGET = a
SRC = mnist.cu
NVCC_FLAGS = \
	-O3 \
	-DNDEBUG \
	-Xptxas=-O3 \
	-arch=sm_89 \
	-maxrregcount=64

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

clean:
	rm $(TARGET)

