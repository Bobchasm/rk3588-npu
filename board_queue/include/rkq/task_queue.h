#pragma once

#include <string>
#include <vector>

namespace rkq {

// 任务状态文件中的逻辑状态。
enum class TaskStatus {
    kQueued,
    kRunning,
    kFinished,
    kFailed,
    kCanceled,
};

// 单个任务的完整描述。
// 第一版队列管理器不关心具体模型框架，只关心“在哪个目录执行什么命令”。
struct TaskRecord {
    std::string task_id;
    std::string user_name;
    std::string host_name;
    std::string submit_time;
    std::string start_time;
    std::string finish_time;
    std::string working_dir;
    std::string command_display;
    std::vector<std::string> command_argv;
    long submit_epoch_ms = 0;
    long start_epoch_ms = 0;
    long finish_epoch_ms = 0;
    int submit_pid = 0;
    int exit_code = 0;
    int cancel_requested = 0;
    TaskStatus status = TaskStatus::kQueued;
};

// 轻量任务队列管理器。
// 设计目标：
// 1. 多人共享单板时，所有占用 NPU 的任务严格串行。
// 2. 用户只需在原命令前面加 `rkq run --`，不用修改各自项目结构。
class TaskQueueManager {
public:
    explicit TaskQueueManager(const std::string& root_dir = "/tmp/rkq");
    ~TaskQueueManager();

    TaskQueueManager(const TaskQueueManager&) = delete;
    TaskQueueManager& operator=(const TaskQueueManager&) = delete;

    int run_command(const std::vector<std::string>& argv);
    int print_status() const;
    int print_current() const;
    int cancel_task(const std::string& task_id);

private:
    std::string root_dir_;
    std::string queue_dir_;
    std::string finished_dir_;
    std::string current_file_;
    std::string history_file_;
    std::string state_lock_file_;
    mutable int state_lock_fd_;

    void ensure_layout() const;
    void lock_state() const;
    void unlock_state() const;

    TaskRecord build_task_record(const std::vector<std::string>& argv) const;
    std::string make_task_file_path(const std::string& task_id) const;
    std::string make_finished_file_path(const std::string& task_id) const;

    void write_task_file(const std::string& path, const TaskRecord& task) const;
    bool read_task_file(const std::string& path, TaskRecord& task) const;
    bool load_current_task(TaskRecord& task) const;
    std::vector<TaskRecord> load_queued_tasks() const;
    void append_history_line(const TaskRecord& task, const std::string& event) const;

    bool try_promote_to_running(const TaskRecord& self);
    void update_current_snapshot(const TaskRecord& task) const;
    void clear_current_snapshot() const;
    void move_task_to_finished(const TaskRecord& task) const;

    int wait_until_turn(const TaskRecord& self) const;
    int execute_task(TaskRecord& self);

    static std::string status_to_string(TaskStatus status);
    static TaskStatus string_to_status(const std::string& text);
    static std::string now_iso8601();
    static long now_epoch_ms();
    static std::string join_command(const std::vector<std::string>& argv);
    static std::string shell_escape(const std::string& value);
    static std::vector<std::string> split_tab_escaped(const std::string& value);
    static std::string join_tab_escaped(const std::vector<std::string>& parts);
};

}  // namespace rkq

