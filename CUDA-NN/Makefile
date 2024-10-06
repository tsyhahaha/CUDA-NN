NVCC = nvcc
NVCCFLAGS = -I ./include

SOURCE_DIR = src
BUILD_DIR = .
EXEC_FILE = main
LOGS_DIR = logs

CPU_SOURCE_FILES := $(shell find $(SOURCE_DIR) -name '*.cpp')
GPU_SOURCE_FILES := $(shell find $(SOURCE_DIR) -name '*.cu')

build: FORCE
	mkdir -p ${BUILD_DIR}
	${NVCC} ${NVCCFLAGS} ${CPU_SOURCE_FILES} ${GPU_SOURCE_FILES} -lineinfo -o ${BUILD_DIR}/${EXEC_FILE}

run:
	mkdir -p ${LOGS_DIR}
	./${BUILD_DIR}/${EXEC_FILE}

clean:
	rm -rf ${BUILD_DIR}

FORCE: