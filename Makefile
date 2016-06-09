CONFIG_FILE := Makefile.config
ifeq ($(wildcard $(CONFIG_FILE)),)
$(error $(CONFIG_FILE) not found. See $(CONFIG_FILE).example.)
endif
include $(CONFIG_FILE)

# Compiler configuration
CXX=g++
CXXFLAGS = -Wall -std=c++11 -fopenmp -Wno-sign-compare -DUSE_OPENCV
CXXDBG = -O0 -g
CXXRUN = -O3

# File dependencies and folders
PROTO = proto
INC = include
SRC = src
BUILD = build
OBJDIR = build/obj

SRC_DIRS := $(shell find $(SRC) -type d -exec bash -c "find {} -maxdepth 1 \
	\( -name '*.cpp' \) | grep -q ." \; -print)

SRCS := $(shell find $(SRC_DIRS) -name '*.cpp' -or -name '*.c' -or -name '*.cc')
INCS := $(shell find $(SRC_DIRS) -name '*.hpp' -or -name '*.h' -or -name '*.hh')

SRCS += $(SRC)/caffetool.pb.cpp
INCS += $(INC)/caffetool.pb.h

RELOBJS := $(sort $(patsubst %.cpp,$(OBJDIR)/rel/%.o,$(SRCS)))
DBGOBJS := $(sort $(patsubst %.cpp,$(OBJDIR)/dbg/%.o,$(SRCS)))


# Includes
INCLUDE = 	-I$(INC) \
			-I$(CAFFE_PATH)/include \
			-I$(CAFFE_PATH)/caffe \
			-I$(CAFFE_PATH)/caffe/src \
			-I$(CAFFE_PATH)/build/src
			
# Library dependencies
ifeq ($(CAFFE_MAKE_BUILD), 1)
	LIBRARY = 	-Wl,-Bstatic,--whole-archive -L$(CAFFE_PATH)/build/lib/ -lcaffe -Wl,-Bdynamic,--no-whole-archive \
				-lopencv_core -lopencv_highgui -lopencv_imgproc \
				-lpthread -lprotobuf -lglog -lgflags -lopenblas \
				-lleveldb -lhdf5_hl -lhdf5 -lsnappy -llmdb -ltiff \
				-lboost_system -lboost_thread -lboost_program_options -lboost_filesystem
else
	LIBRARY = 	-Wl,-Bstatic,--whole-archive -L$(CAFFE_PATH)/build/lib/ -lcaffe -lproto -Wl,-Bdynamic,--no-whole-archive \
				-lopencv_core -lopencv_highgui -lopencv_imgproc \
				-lpthread -lprotobuf -lglog -lgflags -lopenblas \
				-lleveldb -lhdf5_hl -lhdf5 -lsnappy -llmdb -ltiff \
				-lboost_system -lboost_thread -lboost_program_options -lboost_filesystem -lboost_python -lpython2.7
endif

ifeq ($(USE_INDEX_64),1)
	CXXFLAGS += -DUSE_INDEX_64
endif

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
	LIBRARY += -lOpenCL -lrt $(CLLIBS)
endif

ifeq ($(USE_CUDA), 1)
	CXXFLAGS += -DUSE_CUDA
	INCLUDE += -I$(CUDA_DIR)/include
	LIBRARY += -L$(CUDA_DIR)/lib64/ -L$(CUDA_DIR)/lib64/stubs -lcudart -lcublas -lcurand -lnvrtc -lcuda
	ifeq ($(USE_CUDNN), 1)
		LIBRARY += -lcudnn
	endif
endif

# Compiler targets
all: $(BUILD)/caffe_neural_tool $(BUILD)/caffe_neural_tool_dbg
    
$(OBJDIR)/rel/%.o: %.cpp | $(SRC_DIRS) $(INC)/caffetool.pb.h
	@ echo CXX -o $@
	@ mkdir -p $(@D)
	$(Q) $(CXX) $(CXXFLAGS) $(CXXDBG) $(INCLUDE) -c -o $@ $<
    
$(OBJDIR)/dbg/%.o: %.cpp | $(SRC_DIRS) $(INC)/caffetool.pb.h
	@ echo CXX -o $@
	@ mkdir -p $(@D)
	$(Q) $(CXX) $(CXXFLAGS) $(CXXDBG) $(INCLUDE) -c -o $@ $<

$(SRC)/caffetool.pb.cpp $(INC)/caffetool.pb.h: $(PROTO)/caffetool.proto
	protoc --proto_path=$(PROTO) --cpp_out=$(PROTO)/ $(PROTO)/caffetool.proto
	cp $(PROTO)/caffetool.pb.cc $(SRC)/caffetool.pb.cpp
	cp $(PROTO)/caffetool.pb.h $(INC)/caffetool.pb.h

# Run target
$(BUILD)/caffe_neural_tool: $(BUILD) $(CAFFE_PATH)/build/lib/libcaffe.a $(RELOBJS)
	@ echo LD -o $@
	$(Q) $(CXX) $(CXXFLAGS) $(CXXRUN) $(RELOBJS) -o $@ $(LIBRARY)
	
# Debug target
$(BUILD)/caffe_neural_tool_dbg: $(BUILD) $(CAFFE_PATH)/build/lib/libcaffe.a $(DBGOBJS)
	@ echo LD -o $@
	$(Q) $(CXX) $(CXXFLAGS) $(CXXDBG) $(DBGOBJS) -o $@ $(LIBRARY)

# Aux target
$(BUILD):
	mkdir -p $(BUILD)

# Clean target
clean:
	rm -r -f $(BUILD)
	rm -r -f $(PROTO)/caffetool.pb.cc
	rm -r -f $(PROTO)/caffetool.pb.h
	rm -r -f $(SRC)/caffetool.pb.cpp
	rm -r -f $(INC)/caffetool.pb.h
