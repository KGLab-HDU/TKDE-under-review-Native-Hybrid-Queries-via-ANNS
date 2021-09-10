CXX ?= g++
CXXFLAGS += -O3 -march=native -std=c++14 -pthread -fPIC -fopenmp
CXXFLAGS += -I../../include/ -I../../third_party/googletest/googletest/ -I../../third_party/googletest/googletest/include/
CXXFLAGS += -I../../third_party/spdlog/include/ -I../../include/ -I../../third_party/eigen \
			-I../../third_party/boost/assert/include/ -I../../third_party/boost/bind/include/ \
			-I../../third_party/boost/concept_check/include/ -I../../third_party/boost/config/include/ \
			-I../../third_party/boost/core/include/ -I../../third_party/boost/detail/include/ \
			-I../../third_party/boost/heap/include/ -I../../third_party/boost/iterator/include/ \
			-I../../third_party/boost/mp11/include/ -I../../third_party/boost/mpl/include/ \
			-I../../third_party/boost/parameter/include/ -I../../third_party/boost/preprocessor/include/ \
			-I../../third_party/boost/static_assert/include/ -I../../third_party/boost/throw_exception/include/ \
			-I../../third_party/boost/type_traits/include/ -I../../third_party/boost/utility/include/
LDFLAGS += -lpthread -L../../build/lib/static -ln2 -fopenmp

all: search index hybrid_search

index: index.o
	$(CXX) -o $@  $? $(LDFLAGS)

index.o: index.cpp
	$(CXX) $(CXXFLAGS) -c $?

search: search.o
	$(CXX) -o $@  $? $(LDFLAGS)

search.o: search.cpp
	$(CXX) $(CXXFLAGS) -c $?

hybrid_search: hybrid_search.o
	$(CXX) -o $@  $? $(LDFLAGS)

hybrid_search.o: hybrid_search.cpp
	$(CXX) $(CXXFLAGS) -c $?

clean:
	rm -f *.o index search hybrid_search
