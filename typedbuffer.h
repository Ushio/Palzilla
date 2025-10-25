#pragma once

#if (defined(__CUDACC__) || defined(__HIPCC__))
#define TYPED_BUFFER_DEVICE_INLINE __device__ inline
#define TYPED_BUFFER_ASSERT(ExpectTrue) ((void)0)
#else
#include <vector>
#include <Orochi/Orochi.h>
#define TYPED_BUFFER_DEVICE_INLINE
#define TYPED_BUFFER_ASSERT(ExpectTrue) if((ExpectTrue) == 0) { abort(); }
#endif

enum TYPED_BUFFER_TYPE
{
    TYPED_BUFFER_HOST = 0,
    TYPED_BUFFER_DEVICE = 1,
};

template <class T>
struct TypedBuffer
{
    T* m_data = nullptr;
    size_t m_size : 63;
    size_t m_isDevice : 1;

    TYPED_BUFFER_DEVICE_INLINE TypedBuffer(const TypedBuffer&) = delete;
    TYPED_BUFFER_DEVICE_INLINE void operator=(const TypedBuffer&) = delete;

#if (defined(__CUDACC__) || defined(__HIPCC__))
    TYPED_BUFFER_DEVICE_INLINE TypedBuffer() : m_size(0), m_isDevice(TYPED_BUFFER_DEVICE) {}
#else
    void allocate(size_t n)
    {
        if (m_size == n)
        {
            return;
        }

        if (m_isDevice)
        {
            if (m_data) { oroFree((oroDeviceptr)m_data); }
            oroMalloc((oroDeviceptr*)&m_data, n * sizeof(T));
        }
        else
        {
            if (m_data) { free(m_data); }
            m_data = (T*)malloc(n * sizeof(T));
        }
        m_size = n;
    }

    TypedBuffer(TYPED_BUFFER_TYPE type) : m_size(0), m_isDevice(type) {}

    ~TypedBuffer()
    {
        if (m_data)
        {
            if (m_isDevice) { oroFree((oroDeviceptr)m_data); }
            else { free(m_data); }
        }
    }

    TypedBuffer(TypedBuffer<T>&& other)
        : m_data(other.m_data),
          m_size(other.m_size),
          m_isDevice(other.m_isDevice)
    {
        other.m_data = nullptr;
        other.m_size = 0;
    }
#endif

    TYPED_BUFFER_DEVICE_INLINE size_t size() const { return m_size; }

    TYPED_BUFFER_DEVICE_INLINE size_t bytes() const { return m_size * sizeof(T); }

    TYPED_BUFFER_DEVICE_INLINE const T* data() const { return m_data; }

    TYPED_BUFFER_DEVICE_INLINE T* data() { return m_data; }

    TYPED_BUFFER_DEVICE_INLINE const T* begin() const { return data(); }

    TYPED_BUFFER_DEVICE_INLINE const T* end() const { return data() + m_size; }

    TYPED_BUFFER_DEVICE_INLINE T* begin() { return data(); }

    TYPED_BUFFER_DEVICE_INLINE T* end() { return data() + m_size; }

    TYPED_BUFFER_DEVICE_INLINE const T& operator[](int index) const
    {
        return m_data[index];
    }

    TYPED_BUFFER_DEVICE_INLINE T& operator[](int index) { return m_data[index]; }

    TYPED_BUFFER_DEVICE_INLINE bool isDevice() const { return m_isDevice; }

    TYPED_BUFFER_DEVICE_INLINE bool isHost() const { return !isDevice(); }
};

#if (defined(__CUDACC__) || defined(__HIPCC__))
#else
template <class T>
void operator<<(TypedBuffer<T>& to, const TypedBuffer<T>& from )
{
    to.allocate(from.size());
    oroMemcpyKind kind =
        from.isHost()
        ?
        (to.isHost() ? oroMemcpyHostToHost : oroMemcpyHostToDevice)
        :
        (to.isHost() ? oroMemcpyDeviceToHost : oroMemcpyDeviceToDevice);
    oroMemcpy(to.data(), (oroDeviceptr)from.data(), from.size() * sizeof(T), kind);
}

template <class T>
void operator<<(TypedBuffer<T>& to, const std::vector<T>& from)
{
    to.allocate(from.size());
    oroMemcpyKind kind = to.isHost() ? oroMemcpyHostToHost : oroMemcpyHostToDevice;
    oroMemcpy(to.data(), (oroDeviceptr)from.data(), from.size() * sizeof(T), kind);
}

template <class T>
void operator<<(std::vector<T>& to, const TypedBuffer<T>& from)
{
    to.resize(from.size());
    oroMemcpyKind kind = from.isHost() ? oroMemcpyHostToHost : oroMemcpyDeviceToHost;
    oroMemcpy(to.data(), (oroDeviceptr)from.data(), from.size() * sizeof(T), kind);
}
#endif