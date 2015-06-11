CONFIG_FILE := Makefile.config
ifeq ($(wildcard $(CONFIG_FILE)),)
$(error $(CONFIG_FILE) not found. See $(CONFIG_FILE).example.)
endif
include $(CONFIG_FILE)

# Compiler configuration
CXX=g++
CXXFLAGS = -Wall -std=c++11
CXXDBG = -O0 -g
CXXRUN = -O3

# File dependencies and folders
PROTO = proto
SRC = src
INC = include
BUILD = build
FILES := $(shell find $(SRC) -name '*.cpp' -or -name '*.c' -or -name '*.cc')

# Includes
INCLUDE = 	-I$(INC) \
			-I$(CAFFE_PATH)/include \
			-I$(CAFFE_PATH)/caffe \
			-I$(CAFFE_PATH)/caffe/src
			
# Library dependencies	
LIBRARY = 	-Wl,-Bstatic,--whole-archive -L$(CAFFE_PATH)/build/lib/ -lcaffe -Wl,-Bdynamic,--no-whole-archive \
			-lopencv_core -lopencv_highgui -lopencv_imgproc \
			-lpthread -lprotobuf -lglog -lgflags -lopenblas \
			-lleveldb -lhdf5_hl -lhdf5 -lsnappy -llmdb -ltiff \
			-lboost_system -lboost_thread -lboost_program_options -lboost_filesystem \
			-fopenmp
			
ifeq ($(USE_GREENTEA), 1)
	# Find a valid OpenCL library
	ifdef OPENCL_INC
		CLLINC = -I'$(OPENCL_INC)'
	endif
	
	ifdef OPENCL_LIB
		CLLIBS = -L'$(OPENCL_LIB)'
	endif
	
	ifdef OPENCLROOT
		CLLIBS = -L'$(OPENCLROOT)'
	endif
	
	ifdef CUDA_PATH
		CLLIBS = -L'$(CUDA_DIR)/lib/x64'
	endif
	
	ifdef INTELOCLSDKROOT
		CLLIBS = -L'$(INTELOCLSDKROOT)/lib/x64'
	endif
	
	ifdef AMDAPPSDKROOT
		CLLIBS = -L'$(AMDAPPSDKROOT)/lib/x86_64'
		CLLINC = -I'$(AMDAPPSDKROOT)/include'
	endif

	CXXFLAGS += -DUSE_GREENTEA -DVIENNACL_WITH_OPENCL
	INCLUDE += $(CLLINC) -I$(VIENNACL_DIR)
	ifeq ($(USE_CLBLAS), 1)
		LIBRARY += -lclBLAS
	endif
	ifeq ($(USE_VIENNACLBLAS), 1)
		LIBRARY += -lviennacl
	endif
	LIBRARY += -lOpenCL -lrt $(CLLIBS)
endif

ifeq ($(USE_CUDA), 1)
	CXXFLAGS += -DUSE_CUDA
	INCLUDE += -I$(CUDA_DIR)/include
	LIBRARY += -L$(CUDA_DIR)/lib64/ -lcudart -lcublas -lcurand
endif

# Compiler targets
all: caffe_neural_tool caffe_neural_tool_dbg

proto: $(PROTO)/caffetool.proto
	protoc --proto_path=$(PROTO) --cpp_out=$(PROTO)/ $(PROTO)/caffetool.proto
	cp $(PROTO)/caffetool.pb.cc $(SRC)/caffetool.pb.cc
	cp $(PROTO)/caffetool.pb.h $(INC)/caffetool.pb.h

# Run target
caffe_neural_tool: aux proto $(FILES)
	$(CXX) $(CXXFLAGS) $(CXXRUN) $(INCLUDE) -o $(BUILD)/caffe_neural_tool $(FILES) $(LIBRARY)
	
# Debug target
caffe_neural_tool_dbg: aux proto $(FILES)
	$(CXX) $(CXXFLAGS) $(CXXDBG) $(INCLUDE) -o $(BUILD)/caffe_neural_tool_dbg $(FILES) $(LIBRARY)

# Aux target
aux:
	mkdir -p $(BUILD)

# Clean target
clean:
	rm -r $(BUILD)