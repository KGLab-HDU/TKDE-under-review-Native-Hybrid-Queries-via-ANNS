

#pragma once

#include <unistd.h>

namespace n2 {

class Mmap {
public:
    explicit Mmap(char const* fname);
    ~Mmap();
    void Map(char const* fname);
    void UnMap();
    size_t QueryFileSize() const;
    
    inline char* GetData() const { return data_; }
    inline bool IsOpen() const { return file_handle_ != -1; }
    inline int GetFileHandle() const { return file_handle_; }
    inline size_t GetFileSize() const { return file_size_; }

private:
    char* data_ = nullptr;
    size_t file_size_ = 0;
    int file_handle_ = -1;
};

} // namespace n2
