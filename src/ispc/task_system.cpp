#include "velomind/ispc_runtime.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

// 与 ISPC 生成代码约定的任务入口签名保持完全一致。
typedef void (*TaskFuncType)(void* data, int threadIndex, int threadCount,
                             int taskIndex, int taskCount,
                             int taskIndex0, int taskIndex1, int taskIndex2,
                             int taskCount0, int taskCount1, int taskCount2);

// TaskGroup 采用侵入式引用计数管理生命周期，彻底解耦 ISPCSync 与工作线程完成通知的销毁竞争。
struct TaskGroup {
    std::atomic<int> ref_count{1};
    std::vector<void*> allocations;
    std::atomic<int> remaining_tasks{0};
    std::mutex mutex;
    std::condition_variable cv;

    void add_ref(int n = 1) noexcept {
        ref_count.fetch_add(n, std::memory_order_relaxed);
    }

    void release() noexcept {
        if (ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete this;
        }
    }

    ~TaskGroup() {
        for (void* p : allocations) {
            std::free(p);
        }
    }
};

class ThreadPool {
public:
    struct TaskItem {
        TaskGroup* group{nullptr};
        TaskFuncType func{nullptr};
        void* data{nullptr};
        int count0{0};
        int count1{0};
        int count2{0};
        int total_tasks{0};
        std::atomic<int> next_index{0};
    };

    static ThreadPool& instance() {
        static ThreadPool pool;
        return pool;
    }

    ThreadPool() : stop_(false) {
        init_workers(determine_default_threads());
    }

    ~ThreadPool() {
        shutdown();
    }

    std::size_t get_thread_count() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return workers_.empty() ? 1 : (workers_.size() + 1);
    }

    void set_thread_budget(std::size_t count) {
        if (count == 0) {
            count = determine_default_threads();
        }
        std::size_t num_workers = (count <= 1) ? 0 : (count - 1);
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if ((workers_.empty() && num_workers == 0) || (workers_.size() == num_workers)) {
                return;
            }
        }
        shutdown();
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stop_ = false;
            init_workers(count);
        }
    }

    void enqueue(std::shared_ptr<TaskItem> item) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            queue_.push_back(item);
        }
        cv_.notify_all();
    }

    // 工作线程或等待线程取任务执行；preferred_group 优先处理当前同步所属任务。
    bool do_work(int thread_idx = -1, int thread_cnt = -1, TaskGroup* preferred_group = nullptr) {
        if (thread_idx < 0) {
            thread_idx = static_cast<int>(workers_.size());
        }
        if (thread_cnt < 0) {
            thread_cnt = static_cast<int>(workers_.size() + 1);
        }

        std::shared_ptr<TaskItem> item;
        int taskIdx = -1;

        while (true) {
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                if (preferred_group != nullptr) {
                    for (auto it = queue_.begin(); it != queue_.end(); ++it) {
                        if ((*it)->group == preferred_group &&
                            (*it)->next_index.load(std::memory_order_relaxed) < (*it)->total_tasks) {
                            item = *it;
                            break;
                        }
                    }
                }
                if (!item) {
                    while (!queue_.empty()) {
                        auto candidate = queue_.front();
                        if (candidate->next_index.load(std::memory_order_relaxed) >= candidate->total_tasks) {
                            queue_.pop_front();
                        } else {
                            item = candidate;
                            break;
                        }
                    }
                }
            }
            if (!item) return false;

            taskIdx = item->next_index.fetch_add(1, std::memory_order_relaxed);
            if (taskIdx < item->total_tasks) {
                break;
            }
            item.reset();
        }

        int t0 = taskIdx % item->count0;
        int t1 = (taskIdx / item->count0) % item->count1;
        int t2 = taskIdx / (item->count0 * item->count1);
        item->func(item->data, thread_idx, thread_cnt,
                   taskIdx, item->total_tasks,
                   t0, t1, t2,
                   item->count0, item->count1, item->count2);

        TaskGroup* group = item->group;
        if (group->remaining_tasks.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lk(group->mutex);
            group->cv.notify_all();
        }
        group->release();
        return true;
    }

private:
    static unsigned int determine_default_threads() {
        const char* env = std::getenv("VELOMIND_ISPC_THREADS");
        if (!env) env = std::getenv("ISPC_NUM_THREADS");
        if (env) {
            int val = std::atoi(env);
            if (val > 0) return static_cast<unsigned int>(val);
        }
        unsigned int hw = std::thread::hardware_concurrency();
        if (hw == 0) return 4;
        // 默认将线程预算收敛至至多 8 线程，避免多核环境下的细粒度任务争用与缓存颠簸。
        return std::min(hw, 8u);
    }

    void init_workers(std::size_t total_threads) {
        if (total_threads <= 1) return;
        unsigned int n = static_cast<unsigned int>(total_threads - 1);
        workers_.reserve(n);
        for (unsigned int i = 0; i < n; ++i) {
            workers_.emplace_back([this, i, n]() {
                worker_loop(static_cast<int>(i), static_cast<int>(n + 1));
            });
        }
    }

    void shutdown() {
        std::vector<std::thread> workers_to_join;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stop_ = true;
            workers_to_join.swap(workers_);
        }
        cv_.notify_all();
        for (auto& w : workers_to_join) {
            if (w.joinable()) {
                w.join();
            }
        }
    }

    void worker_loop(int thread_idx, int thread_cnt) {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                cv_.wait(lock, [this]() {
                    return stop_ || !queue_.empty();
                });
                if (stop_ && queue_.empty()) return;
            }
            while (do_work(thread_idx, thread_cnt)) {
            }
        }
    }

    std::vector<std::thread> workers_;
    std::deque<std::shared_ptr<TaskItem>> queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable cv_;
    bool stop_{false};
};

}

namespace velomind::backend::ispc {

std::size_t get_thread_count() {
    return ThreadPool::instance().get_thread_count();
}

void set_thread_budget(std::size_t count) {
    ThreadPool::instance().set_thread_budget(count);
}

}

extern "C" {

void* ISPCAlloc(void** handlePtr, int64_t size, int32_t alignment) {
    if (!handlePtr) return nullptr;
    auto* group = static_cast<TaskGroup*>(*handlePtr);
    if (!group) {
        group = new TaskGroup();
        *handlePtr = group;
    }
    // 至少 64 字节对齐，满足 AVX2/AVX-512 向量访问和缓存行边界要求。
    size_t align = alignment < 64 ? 64 : static_cast<size_t>(alignment);
    size_t sz = ((static_cast<size_t>(size) + align - 1) / align) * align;
    void* ptr = nullptr;
    if (posix_memalign(&ptr, align, sz) != 0) {
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(group->mutex);
        group->allocations.push_back(ptr);
    }
    return ptr;
}

void ISPCLaunch(void** handlePtr, void* f, void* data, int count0, int count1, int count2) {
    if (!handlePtr || !f) return;
    auto* group = static_cast<TaskGroup*>(*handlePtr);
    if (!group) {
        group = new TaskGroup();
        *handlePtr = group;
    }
    int total = count0 * count1 * count2;
    if (total <= 0) return;

    // 单任务或单线程预算模式：直接在调用线程内串行执行，消除线程池入队、互斥锁与唤醒开销。
    if (total == 1 || ThreadPool::instance().get_thread_count() <= 1) {
        auto fn = reinterpret_cast<TaskFuncType>(f);
        for (int i = 0; i < total; ++i) {
            int t0 = i % count0;
            int t1 = (i / count0) % count1;
            int t2 = i / (count0 * count1);
            fn(data, 0, 1, i, total, t0, t1, t2, count0, count1, count2);
        }
        return;
    }

    group->add_ref(total);
    group->remaining_tasks.fetch_add(total, std::memory_order_relaxed);

    auto item = std::make_shared<ThreadPool::TaskItem>();
    item->group = group;
    item->func = reinterpret_cast<TaskFuncType>(f);
    item->data = data;
    item->count0 = count0;
    item->count1 = count1;
    item->count2 = count2;
    item->total_tasks = total;
    item->next_index.store(0, std::memory_order_relaxed);

    ThreadPool::instance().enqueue(item);
}

void ISPCSync(void* handle) {
    if (!handle) return;
    auto* group = static_cast<TaskGroup*>(handle);
    while (group->remaining_tasks.load(std::memory_order_acquire) > 0) {
        if (!ThreadPool::instance().do_work(-1, -1, group)) {
            std::unique_lock<std::mutex> lk(group->mutex);
            group->cv.wait_for(lk, std::chrono::microseconds(50), [&]() {
                return group->remaining_tasks.load(std::memory_order_acquire) == 0;
            });
        }
    }
    group->release();
}

}
