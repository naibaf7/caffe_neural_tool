# Paths to define
CAFFE_PATH = ../caffe_gt
CUDA_PATH = $(CUDA_HOME)
VIENNACL_PATH = ../ViennaCL

# Compiler configuration
CXX=g++
CXXFLAGS = -Wall -std=c++11 -DUSE_GREENTEA -DVIENNACL_WITH_OPENCL
CXXDBG = -O0 -g
CXXRUN = -O3

# File dependencies and folders
PROTO = proto
SRC = src
INC = include
BUILD = build
FILES := $(shell find $(SRC) -name '*.cpp' -or -name '*.c' -or -name '*.cc')

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
	CLLIBS = -L'$(CUDA_PATH)/lib/x64'
endif

ifdef INTELOCLSDKROOT
	CLLIBS = -L'$(INTELOCLSDKROOT)/lib/x64'
endif

ifdef AMDAPPSDKROOT
	CLLIBS = -L'$(AMDAPPSDKROOT)/lib/x86_64'
	CLLINC = -I'$(AMDAPPSDKROOT)/include'
endif

# Includes
INCLUDE = 	-I$(INC) \
			-I$(CAFFE_PATH)/include \
			-I$(CAFFE_PATH)/caffe \
			-I$(CAFFE_PATH)/caffe/src \
			$(CLLINC) \
			-I$(VIENNACL_PATH) \
			-I$(CUDA_PATH)/include

# Library dependencies	
LIBRARY = 	-Wl,--whole-archive -l:$(CAFFE_PATH)/build/lib/libcaffe.a -Wl,--no-whole-archive \
			-lopencv_core -lopencv_highgui -lopencv_imgproc \
			-lpthread -lprotobuf -lglog -lgflags -lopenblas \
			-lleveldb -lhdf5_hl -lhdf5 -lsnappy -llmdb -ltiff\
			-lboost_system -lboost_thread -lboost_program_options -lboost_filesystem\
			-fopenmp \
			-lviennacl -lclBLAS -lOpenCL -lrt $(CLLIBS) \
			-L$(CUDA_PATH)/lib64/ -lcudart -lcublas -lcurand
			
			
# Compiler targets
all: aux proto caffe_neural_tool caffe_neural_tool_dbg

proto: $(PROTO)/caffetool.proto
	protoc --proto_path=$(PROTO) --cpp_out=$(PROTO)/ $(PROTO)/caffetool.proto
	cp $(PROTO)/caffetool.pb.cc $(SRC)/caffetool.pb.cc
	cp $(PROTO)/caffetool.pb.h $(INC)/caffetool.pb.h

# Run target
caffe_neural_tool: proto $(FILES)
	$(CXX) $(CXXFLAGS) $(CXXRUN) $(INCLUDE) -o $(BUILD)/caffe_neural_tool $(FILES) $(LIBRARY)
	
# Debug target
caffe_neural_tool_dbg: proto $(FILES)
	$(CXX) $(CXXFLAGS) $(CXXDBG) $(INCLUDE) -o $(BUILD)/caffe_neural_tool_dbg $(FILES) $(LIBRARY)

# Aux target
aux:
	mkdir -p build

# Clean target
clean:
	rm -r $(BUILD)/*